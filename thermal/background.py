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
    body_ir: np.ndarray
    total: np.ndarray
    eclipse: np.ndarray

    def average_total(self):
        return self.total.mean(axis=(1, 2))


def _broadcast_temperature(u, temperature, view):
    """Validate and broadcast a temperature input against a view-factor field."""
    t = np.asarray(temperature, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("temperature must be non-negative")
    if t.ndim == 0:
        return t
    if t.ndim == 1:
        if t.shape[0] != u.shape[0]:
            raise ValueError(
                "1-D temperature must match the orbit grid length"
            )
        t = t[:, None, None]
    try:
        np.broadcast_shapes(t.shape, view.shape)
    except ValueError as exc:
        raise ValueError(
            "temperature must be scalar, 1-D orbit trace, "
            "or broadcastable to the view-factor field"
        ) from exc
    return t


def radiative_background(profile, *,
                         s0=S0,
                         j_ir=J_IR,
                         a_earth=A_ALB,
                         solar_panel_temperature_K=None,
                         solar_panel_emittance=1.0,
                         body_temperature=None,
                         body_emittance=1.0):
    """Convert geometric loading factors into incident radiative background."""
    if not isinstance(profile, SurfaceLoadingProfile):
        raise TypeError("profile must be a SurfaceLoadingProfile instance")
    if solar_panel_emittance < 0.0 or solar_panel_emittance > 1.0:
        raise ValueError("solar_panel_emittance must lie in [0, 1]")
    if body_emittance < 0.0 or body_emittance > 1.0:
        raise ValueError("body_emittance must lie in [0, 1]")

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
        panel_temperature = _broadcast_temperature(
            profile.u, solar_panel_temperature_K, profile.solar_panel_view,
        )
        solar_panel_ir = (
            solar_panel_emittance
            * SIGMA_SB
            * panel_temperature ** 4
            * profile.solar_panel_view
        )

    if body_temperature is None:
        body_ir = np.zeros_like(profile.other_structure_view)
    else:
        body_temp = _broadcast_temperature(
            profile.u, body_temperature, profile.other_structure_view,
        )
        body_ir = (
            body_emittance * SIGMA_SB * body_temp ** 4
            * profile.other_structure_view
        )

    total = earth_ir + albedo + solar + solar_panel_ir + body_ir
    return SurfaceBackgroundProfile(
        surface_name=profile.surface_name,
        u=profile.u,
        width=profile.width,
        height=profile.height,
        earth_ir=earth_ir,
        albedo=albedo,
        solar=solar,
        solar_panel_ir=solar_panel_ir,
        body_ir=body_ir,
        total=total,
        eclipse=profile.eclipse,
    )
