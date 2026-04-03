"""Steady-state thermal balance solver for realized CubeSat surfaces.

This module sits immediately downstream of the thermal background layer.

Pipeline position:
    surface_loading_propagate  ->  radiative_background  ->  steady_state_temperature
                                                         ->  effective_sink_temperature

Both functions take a SurfaceBackgroundProfile (per-patch W/m² orbit traces) and return
a profile dataclass whose temperature arrays have shape (n_time, ny, nx).

Physical model
--------------
At each orbit position and each patch, assume steady-state with zero internal dissipation:

    q_absorbed = α_solar * (q_solar + q_albedo) + ε * (q_earth_ir + q_panel_ir)
    q_emitted  = ε * σ * T^4

    T_ss = (q_absorbed / (ε * σ)) ^ 0.25

The solar and albedo terms use the solar absorptivity α_solar; the thermal IR terms
(Earth emission, panel re-radiation) use the surface IR emissivity ε.

Effective sink temperature
--------------------------
The composite IR environment the surface faces, expressed as an equivalent blackbody
temperature independent of surface material:

    T_sink = ((q_earth_ir + q_panel_ir) / σ) ^ 0.25

This answers: "what temperature sink is the +/-Y radiator looking at?"
Earth IR and solar-panel re-radiation are the warm contributors; deep space (which
contributes ~0 W/m²) is implicitly the cold background.
"""

from dataclasses import dataclass

import numpy as np

from .background import SurfaceBackgroundProfile
from .constants import SIGMA_SB


def _validate_background(background):
    if not isinstance(background, SurfaceBackgroundProfile):
        raise TypeError("background must be a SurfaceBackgroundProfile instance")


@dataclass(frozen=True)
class SurfaceThermalProfile:
    """Steady-state temperature and absorbed flux for one named surface.

    Attributes
    ----------
    surface_name : str
    u : ndarray, shape (n_time,)
        Orbit argument of latitude [rad].
    temperature : ndarray, shape (n_time, ny, nx)
        Steady-state patch temperature [K].
    q_absorbed : ndarray, shape (n_time, ny, nx)
        Net absorbed flux per patch [W/m²].
    eclipse : ndarray, shape (n_time,), dtype bool
    alpha_solar : float
        Solar absorptivity used.
    epsilon : float
        IR emissivity used.
    """
    surface_name: str
    u: np.ndarray
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
    """Effective IR sink temperature the surface faces at each orbit position.

    This is the equivalent blackbody temperature of the combined IR environment
    (Earth emission + solar-panel re-radiation).  It is a property of the
    environment only — independent of the surface's own material.

    Attributes
    ----------
    surface_name : str
    u : ndarray, shape (n_time,)
        Orbit argument of latitude [rad].
    T_sink : ndarray, shape (n_time, ny, nx)
        Effective IR sink temperature [K].
    eclipse : ndarray, shape (n_time,), dtype bool
    """
    surface_name: str
    u: np.ndarray
    T_sink: np.ndarray
    eclipse: np.ndarray

    def average_T_sink(self):
        """Mean sink temperature across all patches at each timestep, shape (n_time,)."""
        return self.T_sink.mean(axis=(1, 2))


def steady_state_temperature(background, *, alpha_solar, epsilon):
    """Compute per-patch steady-state temperature from a radiative background profile.

    Parameters
    ----------
    background : SurfaceBackgroundProfile
    alpha_solar : float
        Solar absorptivity of the surface material.  Range [0, 1].
    epsilon : float
        IR emissivity of the surface material.  Range (0, 1].

    Returns
    -------
    SurfaceThermalProfile
    """
    _validate_background(background)
    if not (0.0 <= alpha_solar <= 1.0):
        raise ValueError("alpha_solar must lie in [0, 1]")
    if not (0.0 < epsilon <= 1.0):
        raise ValueError("epsilon must lie in (0, 1]")

    alpha_solar = float(alpha_solar)
    epsilon = float(epsilon)
    q_absorbed = (
        alpha_solar * (background.solar + background.albedo)
        + epsilon * (background.earth_ir + background.solar_panel_ir)
    )
    # Guard against tiny numerical negatives before the fourth-root.
    q_absorbed = np.maximum(q_absorbed, 0.0)
    temperature = (q_absorbed / (epsilon * SIGMA_SB)) ** 0.25

    return SurfaceThermalProfile(
        surface_name=background.surface_name,
        u=background.u,
        temperature=temperature,
        q_absorbed=q_absorbed,
        eclipse=background.eclipse,
        alpha_solar=alpha_solar,
        epsilon=epsilon,
    )


def effective_sink_temperature(background):
    """Compute the effective IR sink temperature the surface faces.

    The result represents the combined Earth IR and solar-panel re-radiation
    environment expressed as a single equivalent blackbody temperature.  Deep
    space (which contributes ~0 W/m²) is the implicit cold background.

    This is purely environmental — it does not depend on the surface material.

    Parameters
    ----------
    background : SurfaceBackgroundProfile

    Returns
    -------
    SinkTemperatureProfile
    """
    _validate_background(background)

    q_ir = background.earth_ir + background.solar_panel_ir
    q_ir = np.maximum(q_ir, 0.0)
    T_sink = (q_ir / SIGMA_SB) ** 0.25

    return SinkTemperatureProfile(
        surface_name=background.surface_name,
        u=background.u,
        T_sink=T_sink,
        eclipse=background.eclipse,
    )
