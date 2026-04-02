"""Disk-integrated and panel-resolved geometry propagators.

The actively used geometry path resolves the Earth disk in solid angle and
supports both face-level and panel-level loading models. The older scalar
flat-plate approximation is kept in ``geometry.legacy``.
"""

from dataclasses import dataclass

import numpy as np

from .constants import A_ALB, FACES, J_IR, S0
from .earthdisk import (EarthDiskQuadrature, face_coordinates,
                        integrate_face_response)
from .panel import PanelLoadingProfile, RectangularPanel
from .sampling import propagation_grid


@dataclass(frozen=True)
class EarthLoadingProfile:
    """Disk-integrated Earth loading over one orbit."""
    u: np.ndarray
    view: dict
    ir: dict
    albedo: dict
    eclipse: np.ndarray

    def total(self, face):
        """Earth IR + albedo loading for ``face`` in W/m^2."""
        return self.ir[face] + self.albedo[face]


def earth_loading_propagate(orbit, law, n=360, *,
                            face_masks=None,
                            n_mu=24, n_az=72,
                            s0=S0, j_ir=J_IR, a_earth=A_ALB):
    """Sweep argument of latitude and integrate the visible Earth disk."""
    if face_masks is None:
        face_masks = {}
    else:
        unknown = sorted(set(face_masks) - set(FACES))
        if unknown:
            raise ValueError(f"Unknown face mask labels: {unknown}")

    u_arr = propagation_grid(orbit, law, n)
    n_samp = u_arr.size
    view = {face: np.zeros(n_samp) for face in FACES}
    ir = {face: np.zeros(n_samp) for face in FACES}
    albedo = {face: np.zeros(n_samp) for face in FACES}
    eclipse = np.empty(n_samp, dtype=bool)

    quad = EarthDiskQuadrature.build(orbit.rho, n_mu=n_mu, n_az=n_az)
    sun_eci = orbit.sun_eci() if a_earth != 0.0 else None

    for k, uk in enumerate(u_arr):
        rotation = law(uk, orbit)
        nadir_eci = orbit.nadir_eci(uk)
        eclipse[k] = orbit.in_eclipse(uk)

        samples = quad.sample(nadir_eci, orbit.a)
        dirs_body = samples.dirs_eci @ rotation.m
        albedo_weight = (
            samples.surface_solar_cosine(sun_eci) if sun_eci is not None else None
        )

        for face in FACES:
            dirs_face, _, _ = face_coordinates(dirs_body, face)
            mask = face_masks.get(face)

            geom = integrate_face_response(dirs_face, samples.weights, mask=mask)
            view[face][k] = geom
            ir[face][k] = j_ir * geom

            if albedo_weight is not None:
                alb = integrate_face_response(
                    dirs_face,
                    samples.weights,
                    mask=mask,
                    sample_weight=albedo_weight,
                )
                albedo[face][k] = a_earth * s0 * alb

    return EarthLoadingProfile(
        u=u_arr,
        view=view,
        ir=ir,
        albedo=albedo,
        eclipse=eclipse,
    )


def panel_loading_propagate(orbit, law, panel, *, face='+Y', n=180,
                            n_mu=24, n_az=72,
                            s0=S0, j_ir=J_IR, a_earth=A_ALB):
    """Sweep argument of latitude and integrate a patch-resolved panel model."""
    if face not in FACES:
        raise ValueError(f"Unknown face label {face!r}")
    if not isinstance(panel, RectangularPanel):
        raise TypeError("panel must be a RectangularPanel instance")

    u_arr = propagation_grid(orbit, law, n)
    n_samp = u_arr.size
    view = np.zeros((n_samp, panel.ny, panel.nx), dtype=float)
    ir = np.zeros_like(view)
    albedo = np.zeros_like(view)
    eclipse = np.empty(n_samp, dtype=bool)

    quad = EarthDiskQuadrature.build(orbit.rho, n_mu=n_mu, n_az=n_az)
    sun_eci = orbit.sun_eci() if a_earth != 0.0 else None

    for k, uk in enumerate(u_arr):
        rotation = law(uk, orbit)
        nadir_eci = orbit.nadir_eci(uk)
        eclipse[k] = orbit.in_eclipse(uk)

        samples = quad.sample(nadir_eci, orbit.a)
        dirs_body = samples.dirs_eci @ rotation.m
        dirs_face, _, _ = face_coordinates(dirs_body, face)

        geom = panel.integrate(dirs_face, samples.weights)
        view[k] = geom
        ir[k] = j_ir * geom

        if sun_eci is not None:
            alb = panel.integrate(
                dirs_face,
                samples.weights,
                sample_weight=samples.surface_solar_cosine(sun_eci),
            )
            albedo[k] = a_earth * s0 * alb

    return PanelLoadingProfile(
        face=face,
        panel=panel,
        u=u_arr,
        view=view,
        ir=ir,
        albedo=albedo,
        eclipse=eclipse,
    )
