"""Inspection and query utilities for realized CubeSat geometry.

These helpers work with RealizedGeometry and RectSurface objects but do not
produce plots. Use geometry.CubeSat.plots for matplotlib visualizations.
"""

import numpy as np

from .surfaces import RealizedGeometry, _AXIS_MAP


def signed_axis_label(vector, tol=1e-9):
    """Return the body-axis label for a unit vector, e.g. '+Y', or a fallback string."""
    vector = np.asarray(vector, dtype=float)
    for label, axis in _AXIS_MAP:
        if np.allclose(vector, axis, atol=tol):
            return label
    return str(tuple(np.round(vector, 4).tolist()))


def opposite_axis_label(label):
    """Return the label for the axis opposite to *label*, e.g. '+Y' -> '-Y'."""
    if isinstance(label, str) and len(label) == 2 and label[0] in '+-' and label[1] in 'XYZ':
        return ('-' if label[0] == '+' else '+') + label[1]
    return f'-({label})'


def face_frame_labels(surface):
    """Return body-axis labels for the (u, v, normal) axes of *surface*.

    Returns
    -------
    tuple[str, str, str]
        Labels for +u, +v, and normal directions.
    """
    frame = surface.frame_matrix
    return (
        signed_axis_label(frame[:, 0]),
        signed_axis_label(frame[:, 1]),
        signed_axis_label(frame[:, 2]),
    )


def surface_body_role(surface):
    """Return a compact body-role label derived from the realized normal."""
    normal_label = signed_axis_label(surface.normal)
    if 'solar_panel' in surface.tags:
        return f'body {normal_label} solar panel'
    if 'bus' in surface.tags:
        return f'body {normal_label} bus face'
    return f'body {normal_label} surface'


def surface_by_normal(realized, target_normal, *, tag=None, tol=1e-9):
    """Return the unique surface whose normal matches *target_normal*.

    Parameters
    ----------
    realized : RealizedGeometry
    target_normal : array-like (3,)
    tag : str, optional
        Restrict the search to surfaces carrying this tag.
    tol : float
        Absolute tolerance for normal comparison.

    Raises
    ------
    ValueError
        If the match is not unique.
    """
    target = np.asarray(target_normal, dtype=float)
    target = target / np.linalg.norm(target)
    surfaces = realized.surfaces if tag is None else realized.by_tag(tag)
    matches = [s for s in surfaces if np.allclose(s.normal, target, atol=tol)]
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one surface with normal {tuple(target.tolist())}, '
            f'found {len(matches)}'
        )
    return matches[0]


def print_surface_summary(realized):
    """Print a compact table of all surfaces in *realized*."""
    header = (
        f"{'builder id':24s} {'body role':22s} {'center [m]':30s} "
        f"{'normal':8s} {'size [m]':18s} patches tags"
    )
    print(header)
    print('-' * len(header))
    for surface in realized.surfaces:
        center = tuple(np.round(surface.center, 4).tolist())
        normal = signed_axis_label(surface.normal)
        role = surface_body_role(surface)
        size = f"{surface.width:.4f} x {surface.height:.4f}"
        patches = (
            '1 x 1' if surface.patch_shape is None
            else f"{surface.patch_shape[0]} x {surface.patch_shape[1]}"
        )
        tags = ', '.join(surface.tags)
        print(
            f"{surface.name:24s} {role:22s} {str(center):30s} {normal:8s} "
            f"{size:18s} {patches:7s} {tags}"
        )


def print_mounted_role_table(realized):
    """Print a table showing each surface's body-frame axis roles."""
    header = (
        f"{'builder id':24s} {'body role':22s} {'normal':8s} {'+u':8s} "
        f"{'+v':8s} {'center [m]':30s} tags"
    )
    print(header)
    print('-' * len(header))
    for surface in realized.surfaces:
        u_label, v_label, n_label = face_frame_labels(surface)
        center = tuple(np.round(surface.center, 4).tolist())
        role = surface_body_role(surface)
        tags = ', '.join(surface.tags)
        print(
            f"{surface.name:24s} {role:22s} {n_label:8s} {u_label:8s} {v_label:8s} "
            f"{str(center):30s} {tags}"
        )
