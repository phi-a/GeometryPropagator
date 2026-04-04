"""Thermal background conversion from geometric surface profiles."""

from dataclasses import dataclass

import numpy as np

from geometry.constants import A_ALB, J_IR, S0
from viewfactor import SurfaceLoadingProfile

from .constants import SIGMA_SB


@dataclass(frozen=True)
class SurfaceBackgroundProfile:
    """Incident radiative background components for one named surface."""
    surface_name: str
    u: np.ndarray
    width: float
    height: float
    earth_ir: np.ndarray
    albedo: np.ndarray
    solar: np.ndarray
    solar_panel_ir: np.ndarray
    total: np.ndarray
    eclipse: np.ndarray

    def average_total(self):
        return self.total.mean(axis=(1, 2))


def _panel_temperature_field(profile, solar_panel_temperature_K):
    temperatures = np.asarray(solar_panel_temperature_K, dtype=float)
    if np.any(temperatures < 0.0):
        raise ValueError("solar_panel_temperature_K must be non-negative")

    if temperatures.ndim == 0:
        return temperatures
    if temperatures.ndim == 1:
        if temperatures.shape[0] != profile.u.shape[0]:
            raise ValueError(
                "1-D solar_panel_temperature_K must have length len(profile.u)"
            )
        temperatures = temperatures[:, None, None]

    try:
        np.broadcast_shapes(temperatures.shape, profile.solar_panel_view.shape)
    except ValueError as exc:
        raise ValueError(
            "solar_panel_temperature_K must be scalar, length len(profile.u), "
            "or broadcastable to profile.solar_panel_view"
        ) from exc
    return temperatures


def radiative_background(profile, *,
                         s0=S0,
                         j_ir=J_IR,
                         a_earth=A_ALB,
                         solar_panel_temperature_K=None,
                         solar_panel_emittance=1.0):
    """Convert geometric loading factors into incident radiative background."""
    if not isinstance(profile, SurfaceLoadingProfile):
        raise TypeError("profile must be a SurfaceLoadingProfile instance")
    if solar_panel_emittance < 0.0 or solar_panel_emittance > 1.0:
        raise ValueError("solar_panel_emittance must lie in [0, 1]")

    earth_ir = j_ir * profile.earth_view
    albedo = a_earth * s0 * profile.albedo_view
    solar = s0 * profile.solar_view

    warm_view_present = bool(np.any(profile.solar_panel_view > 1e-15))
    if solar_panel_temperature_K is None:
        if warm_view_present:
            raise ValueError(
                "solar_panel_temperature_K is required when solar_panel_view is non-zero"
            )
        solar_panel_ir = np.zeros_like(profile.solar_panel_view)
    else:
        panel_temperature = _panel_temperature_field(profile, solar_panel_temperature_K)
        solar_panel_ir = (
            solar_panel_emittance
            * SIGMA_SB
            * panel_temperature ** 4
            * profile.solar_panel_view
        )

    total = earth_ir + albedo + solar + solar_panel_ir
    return SurfaceBackgroundProfile(
        surface_name=profile.surface_name,
        u=profile.u,
        width=profile.width,
        height=profile.height,
        earth_ir=earth_ir,
        albedo=albedo,
        solar=solar,
        solar_panel_ir=solar_panel_ir,
        total=total,
        eclipse=profile.eclipse,
    )
