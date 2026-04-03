"""Spacecraft self-occlusion helpers for realized body geometry.

This module bridges the realized rectangular-surface spacecraft model with
directional view-factor integration. The core operation is:

    for each source patch center and sampled direction,
    cast a ray against the realized spacecraft surfaces and decide whether
    that direction is blocked by local structure.

The result is a patch-by-ray visibility mask that can be multiplied with the
existing cosine-weighted directional kernels in the view-factor layer.
"""

from __future__ import annotations

import math

import numpy as np

from .CubeSat.surfaces import RealizedGeometry, RectSurface


def _resolve_source_surface(realized: RealizedGeometry,
                            source: str | RectSurface) -> RectSurface:
    if isinstance(source, str):
        return realized.by_name(source)
    if isinstance(source, RectSurface):
        return source
    raise TypeError("source must be a surface name or a RectSurface")


def _validate_dirs_body(dirs_body) -> np.ndarray:
    dirs_body = np.asarray(dirs_body, dtype=float)
    if dirs_body.ndim != 2 or dirs_body.shape[1] != 3:
        raise ValueError("dirs_body must have shape (n_rays, 3)")
    return dirs_body


def spacecraft_occlusion_mask(realized: RealizedGeometry,
                              source: str | RectSurface,
                              dirs_body,
                              *,
                              exclude=(),
                              eps: float = 1e-9) -> np.ndarray:
    """Return a patch-by-ray self-occlusion mask for a realized surface.

    Parameters
    ----------
    realized : RealizedGeometry
        Realized spacecraft geometry in the body frame.
    source : str | RectSurface
        Source surface, either by name or as the realized surface object.
    dirs_body : array-like, shape (n_rays, 3)
        Sampled ray directions expressed in the body frame.
    exclude : iterable[str], optional
        Additional surface names to ignore during ray intersection.
        The source surface itself is always excluded.
    eps : float, optional
        Small origin offset along the emitting side of the source patch to
        avoid self-hits against the source plane.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_patches, n_rays). ``True`` means the ray is
        unobstructed by spacecraft structure for that patch.
    """
    if not isinstance(realized, RealizedGeometry):
        raise TypeError("realized must be a RealizedGeometry instance")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    dirs_body = _validate_dirs_body(dirs_body)
    surface = _resolve_source_surface(realized, source)

    centers = surface.patch_centers().reshape(-1, 3)
    normals = surface.patch_normals().reshape(-1, 3)
    n_patches = centers.shape[0]
    n_rays = dirs_body.shape[0]
    visible = np.zeros((n_patches, n_rays), dtype=bool)

    ignored = set(exclude)
    ignored.add(surface.name)

    for patch_index, (center, normal) in enumerate(zip(centers, normals)):
        dots = dirs_body @ normal

        for ray_index, dot in enumerate(dots):
            if surface.two_sided:
                if abs(dot) <= 1e-15:
                    continue
                side = 1.0 if dot > 0.0 else -1.0
                origin = center + side * eps * normal
            else:
                if dot <= 0.0:
                    continue
                origin = center + eps * normal

            hit = realized.first_intersection(
                origin,
                dirs_body[ray_index],
                exclude=tuple(ignored),
            )
            visible[patch_index, ray_index] = hit is None

    return visible


def integrate_surface_response(realized: RealizedGeometry,
                               source: str | RectSurface,
                               dirs_body,
                               solid_angle_weights,
                               *,
                               sample_weight=None,
                               exclude=(),
                               eps: float = 1e-9) -> np.ndarray:
    """Integrate a patch-resolved surface response with self-occlusion.

    This is the rectangular-surface analogue of the existing panel-level
    directional integration, but with a global spacecraft blockage mask.
    """
    dirs_body = _validate_dirs_body(dirs_body)
    solid_angle_weights = np.asarray(solid_angle_weights, dtype=float)
    if solid_angle_weights.ndim != 1 or solid_angle_weights.shape[0] != dirs_body.shape[0]:
        raise ValueError("solid_angle_weights must have shape (n_rays,)")

    surface = _resolve_source_surface(realized, source)
    visibility = spacecraft_occlusion_mask(
        realized,
        surface,
        dirs_body,
        exclude=exclude,
        eps=eps,
    ).astype(float)

    cosine = dirs_body @ surface.normal
    if surface.two_sided:
        cosine = np.abs(cosine)
    else:
        cosine = np.clip(cosine, 0.0, None)

    kernel = visibility * cosine[None, :]

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1 or sample_weight.shape[0] != dirs_body.shape[0]:
            raise ValueError("sample_weight must have shape (n_rays,)")
        kernel = kernel * sample_weight[None, :]

    values = np.sum(kernel * solid_angle_weights[None, :], axis=1) / math.pi
    ny, nx = surface.patch_centers().shape[:2]
    return values.reshape(ny, nx)
