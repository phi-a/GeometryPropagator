"""Legacy scalar view-factor models.

This module retains the original infinitesimal flat-plate approximation for
backward compatibility and reference. The active geometry workflow now lives
in the disk-integrated and panel-resolved propagators.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..constants import A_ALB, FACES, J_IR, S0
from ..sampling import propagation_grid


def earth_vf(cos_alpha, H):
    """Earth view factor from a differential flat plate."""
    alpha_lim = math.acos(1.0 / H)
    alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))

    if alpha <= alpha_lim:
        return cos_alpha / (H * H)

    if alpha >= math.pi - alpha_lim:
        return 0.0

    x = math.sqrt(H * H - 1.0)
    sa = math.sin(alpha)
    ca = cos_alpha
    y = max(-1.0, min(1.0, -x * ca / sa))
    s = math.sqrt(max(0.0, 1.0 - y * y))
    f_earth = ((ca * math.acos(y) + x * sa * s) / (math.pi * H * H)
               + math.atan2(sa * s, x) / math.pi)
    return max(0.0, f_earth)


@dataclass(frozen=True)
class ViewFactorProfile:
    """Legacy scalar view-factor profiles over one orbit."""
    u: np.ndarray
    earth: dict
    eclipse: np.ndarray


@dataclass(frozen=True)
class ThermalProfile:
    """Legacy scalar incident flux profiles over one orbit."""
    u: np.ndarray
    solar: dict
    ir: dict
    albedo: dict
    eclipse: np.ndarray

    def total(self, face):
        return self.solar[face] + self.ir[face] + self.albedo[face]


def propagate(orbit, law, n=360):
    """Sweep argument of latitude with the legacy scalar Earth factor."""
    u_arr = propagation_grid(orbit, law, n)
    n_samp = u_arr.size
    earth = {f: np.empty(n_samp) for f in FACES}
    eclipse = np.empty(n_samp, dtype=bool)
    H = orbit.H

    for k, uk in enumerate(u_arr):
        rotation = law(uk, orbit)
        nadir_eci = orbit.nadir_eci(uk)
        eclipse[k] = orbit.in_eclipse(uk)
        for face, n_body in FACES.items():
            n_eci = rotation.apply(n_body)
            earth[face][k] = earth_vf(float(np.dot(n_eci, nadir_eci)), H)

    return ViewFactorProfile(u=u_arr, earth=earth, eclipse=eclipse)


def thermal_propagate(orbit, law, n=360,
                      s0=S0, j_ir=J_IR, a_earth=A_ALB):
    """Sweep argument of latitude with the legacy scalar thermal model."""
    u_arr = propagation_grid(orbit, law, n)
    n_samp = u_arr.size
    solar = {f: np.zeros(n_samp) for f in FACES}
    ir = {f: np.zeros(n_samp) for f in FACES}
    albedo = {f: np.zeros(n_samp) for f in FACES}
    eclipse = np.empty(n_samp, dtype=bool)

    H = orbit.H
    sun_eci = orbit.sun_eci()

    for k, uk in enumerate(u_arr):
        rotation = law(uk, orbit)
        nadir_eci = orbit.nadir_eci(uk)
        r_hat_eci = -nadir_eci
        eclipse[k] = orbit.in_eclipse(uk)
        cos_illum = max(0.0, float(np.dot(r_hat_eci, sun_eci)))

        for face, n_body in FACES.items():
            n_eci = rotation.apply(n_body)
            cos_nadir = float(np.dot(n_eci, nadir_eci))
            cos_sun = float(np.dot(n_eci, sun_eci))
            f_earth = earth_vf(cos_nadir, H)

            if not eclipse[k]:
                solar[face][k] = s0 * max(0.0, cos_sun)
            ir[face][k] = j_ir * f_earth
            albedo[face][k] = a_earth * s0 * f_earth * cos_illum

    return ThermalProfile(
        u=u_arr,
        solar=solar,
        ir=ir,
        albedo=albedo,
        eclipse=eclipse,
    )
