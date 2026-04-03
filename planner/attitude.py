"""Spacecraft pointing and Earth view factors.

Two-vector triads map body axes into the LVLH frame (R, T, W)
for sunlit and umbra pointing modes.  Earth view factors follow
from the angle between each body face normal and nadir.
"""

import math

from constants import R_E


# ── Vector helpers ────────────────────────────────────────────────

def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _normalize(v):
    m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0] / m, v[1] / m, v[2] / m)


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


# ── LVLH direction ───────────────────────────────────────────────

NADIR = (-1.0, 0.0, 0.0)
_ORBIT_NORMAL = (0.0, 0.0, 1.0)


def _secondary_ref(primary):
    """Nadir-based secondary; falls back to orbit normal if parallel."""
    c = _cross(NADIR, primary)
    if c[0]**2 + c[1]**2 + c[2]**2 > 1e-12:
        return _normalize(c)
    return _normalize(_cross(_ORBIT_NORMAL, primary))


def direction_lvlh(beta: float, uc: float, u: float) -> tuple[float, float, float]:
    """Unit vector in LVLH for a direction with beta angle and culmination.

    Parameters
    ----------
    beta : float
        Beta angle of the direction (rad).
    uc : float
        Argument of latitude of culmination (rad).
    u : float
        Spacecraft argument of latitude (rad).
    """
    cb = math.cos(beta)
    du = u - uc
    return (cb * math.cos(du), -cb * math.sin(du), math.sin(beta))


# ── Two-vector triads ────────────────────────────────────────────

Axes = tuple[tuple[float, float, float], ...]


def sunlit_axes(sun_lvlh: tuple) -> Axes:
    """Body axes (x, y, z) in LVLH for sunlit pointing.

    Primary  : +Z_body → Sun.
    Secondary: +Y_body ⊥ nadir (radiators edge-on to Earth).
    """
    z = _normalize(sun_lvlh)
    y = _secondary_ref(z)
    x = _cross(y, z)
    return x, y, z


def umbra_axes(target_lvlh: tuple) -> Axes:
    """Body axes (x, y, z) in LVLH for umbra pointing.

    Primary  : +X_body → target.
    Secondary: +Y_body ⊥ nadir (radiators edge-on to Earth).
    """
    x = _normalize(target_lvlh)
    y = _secondary_ref(x)
    z = _cross(x, y)
    return x, y, z


# ── Inertial drift ────────────────────────────────────────────────

def drift_axes(axes: Axes, du: float) -> Axes:
    """Rotate body axes for LVLH drift about orbit normal.

    The body frame is inertially fixed (tracking a celestial target).
    As the spacecraft advances by *du* radians in argument of latitude,
    the LVLH frame rotates underneath.  This applies the apparent
    rotation of the body axes in the new LVLH frame.
    """
    c, s = math.cos(du), math.sin(du)
    return tuple(
        (v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2])
        for v in axes
    )


# ── Earth view factor ────────────────────────────────────────────

def earth_view_factor(normal_lvlh: tuple, a: float) -> float:
    """View factor from a differential flat plate to a diffuse sphere.

    Piecewise analytical model (Siegel & Howell, 6th ed.):

    * Full view  (α ≤ α_lim):         F = cos α / H²
    * Partial    (α_lim < α < π-α_lim): horizon-clipped integral
    * No view   (α ≥ π - α_lim):      F = 0

    Parameters
    ----------
    normal_lvlh : tuple
        Outward face normal unit vector in LVLH.
    a : float
        Semi-major axis (m), i.e. distance from Earth centre.
    """
    H = a / R_E
    alpha_lim = math.acos(1.0 / H)

    cos_alpha = _dot(normal_lvlh, NADIR)
    alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))

    if alpha <= alpha_lim:
        return cos_alpha / (H * H)

    if alpha >= math.pi - alpha_lim:
        return 0.0

    # Partial view — horizon cuts across the plate's FOV
    x = math.sqrt(H * H - 1.0)
    sin_a = math.sin(alpha)
    cos_a = cos_alpha
    y = max(-1.0, min(1.0, -x * cos_a / sin_a))
    s = math.sqrt(max(0.0, 1.0 - y * y))

    F = (cos_a * math.acos(y) + x * sin_a * s) / (math.pi * H * H) \
        + math.atan2(sin_a * s, x) / math.pi
    return max(0.0, F)


def all_view_factors(a: float, axes: Axes) -> dict[str, float]:
    """Earth view factor for the six body faces.

    Returns dict keyed by face label (+X, -X, +Y, -Y, +Z, -Z).
    """
    result = {}
    for k, label in enumerate(("X", "Y", "Z")):
        axis = axes[k]
        neg = (-axis[0], -axis[1], -axis[2])
        result[f"+{label}"] = earth_view_factor(axis, a)
        result[f"-{label}"] = earth_view_factor(neg, a)
    return result
