"""Thermal balance solvers for realized CubeSat surfaces.

This module sits immediately downstream of the thermal background layer.

Pipeline position:
    surface_loading_propagate  ->  radiative_background
                                 ->  steady_state_temperature
                                 ->  steady_state_temperature_two_sided
                                 ->  transient_temperature
                                 ->  effective_sink_temperature
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .background import SurfaceBackgroundProfile
from .constants import SIGMA_SB


def _validate_background(background):
    if not isinstance(background, SurfaceBackgroundProfile):
        raise TypeError("background must be a SurfaceBackgroundProfile instance")


def _validate_material_properties(*, alpha_solar, epsilon):
    if not (0.0 <= alpha_solar <= 1.0):
        raise ValueError("alpha_solar must lie in [0, 1]")
    if not (0.0 < epsilon <= 1.0):
        raise ValueError("epsilon must lie in (0, 1]")
    return float(alpha_solar), float(epsilon)


def _absorbed_flux(background, *, alpha_solar, epsilon):
    return (
        alpha_solar * (background.solar + background.albedo)
        + epsilon * (background.earth_ir + background.solar_panel_ir
                     + background.body_ir)
    )


def _validate_paired_backgrounds(front, back):
    _validate_background(front)
    _validate_background(back)

    if front.total.shape != back.total.shape:
        raise ValueError("front and back backgrounds must share the same data shape")
    if front.u.shape != back.u.shape or not np.allclose(front.u, back.u):
        raise ValueError("front and back backgrounds must share the same u samples")
    if front.eclipse.shape != back.eclipse.shape or not np.array_equal(front.eclipse, back.eclipse):
        raise ValueError("front and back backgrounds must share the same eclipse mask")
    if not math.isclose(float(front.width), float(back.width), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("front and back backgrounds must share the same width")
    if not math.isclose(float(front.height), float(back.height), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("front and back backgrounds must share the same height")


def _align_back_to_front(array):
    """Map a back-face patch field onto the front-face patch indexing."""
    return np.flip(np.asarray(array, dtype=float), axis=1)


def _two_sided_absorbed_flux(bg_front, bg_back, *,
                             alpha_front, epsilon_front,
                             alpha_back, epsilon_back):
    q_front = _absorbed_flux(
        bg_front,
        alpha_solar=alpha_front,
        epsilon=epsilon_front,
    )
    q_back = _align_back_to_front(_absorbed_flux(
        bg_back,
        alpha_solar=alpha_back,
        epsilon=epsilon_back,
    ))
    return q_front + q_back


@dataclass(frozen=True)
class SurfaceThermalProfile:
    """Temperature and absorbed flux for one named surface."""
    surface_name: str
    u: np.ndarray
    width: float
    height: float
    temperature: np.ndarray
    q_absorbed: np.ndarray
    eclipse: np.ndarray
    alpha_solar: float
    epsilon: float

    def average_temperature(self):
        """Mean temperature across all patches at each timestep, shape (n_time,)."""
        return self.temperature.mean(axis=(1, 2))

    def peak_temperature(self):
        """Maximum patch temperature at each timestep, shape (n_time,)."""
        return self.temperature.max(axis=(1, 2))

    def min_temperature(self):
        """Minimum patch temperature at each timestep, shape (n_time,)."""
        return self.temperature.min(axis=(1, 2))


@dataclass(frozen=True)
class SinkTemperatureProfile:
    """Effective IR sink temperature the surface faces at each orbit position."""
    surface_name: str
    u: np.ndarray
    width: float
    height: float
    T_sink: np.ndarray
    eclipse: np.ndarray

    def average_T_sink(self):
        """Mean sink temperature across all patches at each timestep, shape (n_time,)."""
        return self.T_sink.mean(axis=(1, 2))


@dataclass(frozen=True)
class ShroudProfile:
    """Environment and equivalent shroud temperature for a named surface.

    T_env    : (q_total / sigma)^0.25  — complete incident flux environment temperature (all sources, material-independent)
    T_shroud : (mean(q_abs) / (eps * sigma))^0.25  — spatially-uniform equivalent shroud temperature
    """
    surface_name: str
    u: np.ndarray
    width: float
    height: float
    T_env: np.ndarray
    T_shroud: np.ndarray
    eclipse: np.ndarray
    alpha_solar: float
    epsilon: float

    def scalar(self, field='T_shroud', mode='radiative'):
        """Reduce patch grid to one representative scalar per timestep.

        Modes
        -----
        radiative : (mean(T**4))^0.25 — radiation-power-equivalent
        mean      : arithmetic mean across patches
        peak      : spatial maximum
        """
        data = getattr(self, field)
        if mode == 'radiative':
            return np.power(np.mean(data ** 4, axis=(1, 2)), 0.25)
        if mode == 'mean':
            return data.mean(axis=(1, 2))
        if mode == 'peak':
            return data.max(axis=(1, 2))
        raise ValueError(f"unknown mode {mode!r}")


def steady_state_temperature(background, *, alpha_solar, epsilon):
    """Compute per-patch steady-state temperature from a radiative background profile."""
    _validate_background(background)
    alpha_solar, epsilon = _validate_material_properties(
        alpha_solar=alpha_solar,
        epsilon=epsilon,
    )

    q_absorbed = np.maximum(
        _absorbed_flux(background, alpha_solar=alpha_solar, epsilon=epsilon),
        0.0,
    )
    temperature = (q_absorbed / (epsilon * SIGMA_SB)) ** 0.25

    return SurfaceThermalProfile(
        surface_name=background.surface_name,
        u=background.u,
        width=background.width,
        height=background.height,
        temperature=temperature,
        q_absorbed=q_absorbed,
        eclipse=background.eclipse,
        alpha_solar=alpha_solar,
        epsilon=epsilon,
    )


def steady_state_temperature_two_sided(bg_front, bg_back, *,
                                       alpha_front, epsilon_front,
                                       alpha_back, epsilon_back):
    """Compute a shared-temperature two-sided steady-state panel solution."""
    _validate_paired_backgrounds(bg_front, bg_back)
    alpha_front, epsilon_front = _validate_material_properties(
        alpha_solar=alpha_front,
        epsilon=epsilon_front,
    )
    alpha_back, epsilon_back = _validate_material_properties(
        alpha_solar=alpha_back,
        epsilon=epsilon_back,
    )

    q_absorbed = np.maximum(
        _two_sided_absorbed_flux(
            bg_front,
            bg_back,
            alpha_front=alpha_front,
            epsilon_front=epsilon_front,
            alpha_back=alpha_back,
            epsilon_back=epsilon_back,
        ),
        0.0,
    )
    epsilon_total = epsilon_front + epsilon_back
    temperature = (q_absorbed / (epsilon_total * SIGMA_SB)) ** 0.25

    return SurfaceThermalProfile(
        surface_name=bg_front.surface_name,
        u=bg_front.u,
        width=bg_front.width,
        height=bg_front.height,
        temperature=temperature,
        q_absorbed=q_absorbed,
        eclipse=bg_front.eclipse,
        alpha_solar=float('nan'),
        epsilon=epsilon_total,
    )


def transient_temperature(bg_front, bg_back, *,
                          alpha_front, epsilon_front,
                          alpha_back, epsilon_back,
                          thermal_capacitance,
                          orbit_period,
                          n_orbits=10,
                          tol=0.5):
    """Integrate a two-sided panel temperature history to periodic steady state."""
    _validate_paired_backgrounds(bg_front, bg_back)
    alpha_front, epsilon_front = _validate_material_properties(
        alpha_solar=alpha_front,
        epsilon=epsilon_front,
    )
    alpha_back, epsilon_back = _validate_material_properties(
        alpha_solar=alpha_back,
        epsilon=epsilon_back,
    )
    thermal_capacitance = float(thermal_capacitance)
    orbit_period = float(orbit_period)
    n_orbits = int(n_orbits)
    tol = float(tol)
    if thermal_capacitance <= 0.0:
        raise ValueError("thermal_capacitance must be positive")
    if orbit_period <= 0.0:
        raise ValueError("orbit_period must be positive")
    if n_orbits <= 0:
        raise ValueError("n_orbits must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    q_absorbed = np.maximum(
        _two_sided_absorbed_flux(
            bg_front,
            bg_back,
            alpha_front=alpha_front,
            epsilon_front=epsilon_front,
            alpha_back=alpha_back,
            epsilon_back=epsilon_back,
        ),
        0.0,
    )
    epsilon_total = epsilon_front + epsilon_back
    n_time, ny, nx = q_absorbed.shape

    time_samples = np.asarray(bg_front.u, dtype=float) / (2.0 * math.pi) * orbit_period
    q_periodic = np.concatenate([q_absorbed, q_absorbed[:1]], axis=0)
    t_periodic = np.concatenate([time_samples, [orbit_period]])
    forcing = interp1d(
        t_periodic,
        q_periodic,
        axis=0,
        kind='linear',
        assume_sorted=True,
        bounds_error=False,
        fill_value=(q_periodic[0], q_periodic[-1]),
    )

    steady_initial = steady_state_temperature_two_sided(
        bg_front,
        bg_back,
        alpha_front=alpha_front,
        epsilon_front=epsilon_front,
        alpha_back=alpha_back,
        epsilon_back=epsilon_back,
    )
    y0 = steady_initial.temperature[0].reshape(-1)
    t_eval = np.concatenate([time_samples, [orbit_period]])

    def rhs(t, y):
        temperature = np.maximum(y, 0.0)
        q_now = np.asarray(forcing(t), dtype=float).reshape(-1)
        q_emit = epsilon_total * SIGMA_SB * temperature ** 4
        return (q_now - q_emit) / thermal_capacitance

    previous_orbit = None
    current_orbit = None
    final_delta = None
    for _ in range(n_orbits):
        solution = solve_ivp(
            rhs,
            (0.0, orbit_period),
            y0,
            t_eval=t_eval,
            vectorized=False,
            rtol=1e-6,
            atol=1e-2,
        )
        if not solution.success:
            raise RuntimeError(f"transient_temperature integration failed: {solution.message}")

        sampled = solution.y[:, :n_time].T.reshape(n_time, ny, nx)
        y0 = solution.y[:, -1]
        current_orbit = sampled

        if previous_orbit is not None:
            final_delta = float(np.max(np.abs(current_orbit - previous_orbit)))
            if final_delta < tol:
                break
        previous_orbit = current_orbit
    else:
        if final_delta is None:
            final_delta = float('nan')
        warnings.warn(
            f"transient_temperature did not converge within {n_orbits} orbits; "
            f"final orbit-to-orbit max delta = {final_delta:.3f} K",
            RuntimeWarning,
            stacklevel=2,
        )

    return SurfaceThermalProfile(
        surface_name=bg_front.surface_name,
        u=bg_front.u,
        width=bg_front.width,
        height=bg_front.height,
        temperature=current_orbit,
        q_absorbed=q_absorbed,
        eclipse=bg_front.eclipse,
        alpha_solar=float('nan'),
        epsilon=epsilon_total,
    )


def effective_sink_temperature(background):
    """Compute the effective IR sink temperature the surface faces."""
    _validate_background(background)

    q_ir = np.maximum(background.earth_ir + background.solar_panel_ir, 0.0)
    T_sink = (q_ir / SIGMA_SB) ** 0.25

    return SinkTemperatureProfile(
        surface_name=background.surface_name,
        u=background.u,
        width=background.width,
        height=background.height,
        T_sink=T_sink,
        eclipse=background.eclipse,
    )


def shroud_temperature(background, *, alpha_solar, epsilon):
    """Compute environment and equivalent shroud temperatures.

    T_env is the complete incident flux environment temperature (all sources:
    earth IR, albedo, solar, solar panel IR, body IR).  Material-independent.
    T_shroud is the spatially-uniform equivalent shroud temperature — the single
    temperature to set TVAC shroud walls to, matching the spatially-averaged
    absorbed flux for a surface with given alpha and epsilon.
    """
    _validate_background(background)
    alpha_solar, epsilon = _validate_material_properties(
        alpha_solar=alpha_solar, epsilon=epsilon,
    )

    q_total = np.maximum(
        background.earth_ir + background.albedo + background.solar
        + background.solar_panel_ir + background.body_ir,
        0.0,
    )
    q_absorbed = np.maximum(
        _absorbed_flux(background, alpha_solar=alpha_solar, epsilon=epsilon),
        0.0,
    )
    q_abs_mean = q_absorbed.mean(axis=(1, 2), keepdims=True)

    T_env = (q_total / SIGMA_SB) ** 0.25
    T_shroud = (q_abs_mean / (epsilon * SIGMA_SB)) ** 0.25

    return ShroudProfile(
        surface_name=background.surface_name,
        u=background.u,
        width=background.width,
        height=background.height,
        T_env=T_env,
        T_shroud=T_shroud,
        eclipse=background.eclipse,
        alpha_solar=alpha_solar,
        epsilon=epsilon,
    )
