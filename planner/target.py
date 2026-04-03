"""Inertial target visibility: FOV clear-sky budget within eclipse."""

import math
from dataclasses import dataclass

from constants import R_E, SGR_A_RA_DEG, SGR_A_DEC_DEG
from orbit import eclipse_half_angle, mean_motion

# ---------------------------------------------------------------------------
# Geometric visibility states
# ---------------------------------------------------------------------------
# Four mutually exclusive conditions based purely on arc membership:
#
#   SUNLIT     — satellite outside the umbra arc (no eclipse)
#   OCCULTED   — in eclipse; target centre behind the Earth limb
#   PARTIAL    — in eclipse; target visible but FOV intersects the Earth limb
#   OBSERVABLE — in eclipse; entire FOV clears the Earth limb
#
SUNLIT     = "sunlit"
OCCULTED   = "occulted"
PARTIAL    = "partial"
OBSERVABLE = "observable"


@dataclass(frozen=True)
class Target:
    """Inertial celestial target in J2000 equatorial coordinates."""
    ra_rad: float
    dec_rad: float
    name: str = ""


# Galactic center (Sgr A*) J2000
GALACTIC_CENTER = Target(
    ra_rad=math.radians(SGR_A_RA_DEG),
    dec_rad=math.radians(SGR_A_DEC_DEG),
    name="Galactic Center",
)


def target_beta_uc(
    i: float, omega: float, target: Target
) -> tuple[float, float]:
    """Beta angle and culmination argument-of-latitude for an inertial target.

    Parameters
    ----------
    i : float
        Orbit inclination (rad).
    omega : float
        RAAN (rad).
    target : Target
        Celestial target coordinates.

    Returns
    -------
    beta : float
        Target beta angle (rad), in [-π/2, π/2].
    uc : float
        Argument of latitude of target culmination (rad).
    """
    sd, cd = math.sin(target.dec_rad), math.cos(target.dec_rad)
    si, ci = math.sin(i), math.cos(i)
    d_ra = omega - target.ra_rad

    sin_beta = si * cd * math.sin(d_ra) + ci * sd
    beta = math.asin(max(-1.0, min(1.0, sin_beta)))

    g_x = cd * math.cos(target.ra_rad - omega)
    g_y = ci * cd * math.sin(target.ra_rad - omega) + si * sd
    uc = math.atan2(g_y, g_x)

    return beta, uc


def clear_half_angle(
    a: float, beta_target: float, fov_half: float
) -> float:
    """Half-angle of the arc where the target FOV clears the Earth limb.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    beta_target : float
        Target beta angle (rad).
    fov_half : float
        Half-cone angle of the instrument FOV (rad).

    Returns
    -------
    lam : float
        Half-angle (rad) of the clear-sky arc, in [0, π].
    """
    # Minimum elevation for the FOV edge to clear the limb
    e_limb = -math.acos(R_E / a)
    e_req = e_limb + fov_half

    cos_beta = math.cos(beta_target)
    if cos_beta == 0.0:
        # Target exactly in orbit plane — degenerate
        return 0.0

    ratio = math.sin(e_req) / cos_beta
    if ratio <= -1.0:
        return math.pi  # always clear
    if ratio >= 1.0:
        return 0.0       # never clear
    return math.acos(ratio)


def arc_intersection(
    c1: float, h1: float, c2: float, h2: float
) -> float:
    """Angular overlap (rad) of two arcs on a circle.

    Each arc is defined by a center and a half-width.  The arcs wrap
    at 2π.  Returns the overlap angle in [0, 2π].

    Parameters
    ----------
    c1, h1 : float
        Center and half-width of arc 1 (rad).
    c2, h2 : float
        Center and half-width of arc 2 (rad).
    """
    if h1 >= math.pi:
        return 2.0 * h2
    if h2 >= math.pi:
        return 2.0 * h1

    # Angular distance between centers, wrapped to [0, π]
    d = abs(math.remainder(c1 - c2, 2 * math.pi))

    if d >= h1 + h2:
        return 0.0            # disjoint
    if d <= abs(h1 - h2):
        return 2.0 * min(h1, h2)  # one contained in the other
    return h1 + h2 - d        # partial overlap


def open_sky_budget(
    a: float,
    i: float,
    omega: float,
    target: Target,
    fov_half: float,
    beta_sun: float,
    uc_sun: float,
) -> float:
    """Usable science time (s): intersection of umbra and target-clear arcs.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    i : float
        Inclination (rad).
    omega : float
        RAAN (rad).
    target : Target
        Celestial target.
    fov_half : float
        Instrument half-cone FOV (rad).
    beta_sun : float
        Solar beta angle (rad), pre-computed.
    uc_sun : float
        Solar culmination argument of latitude (rad), pre-computed.

    Returns
    -------
    float
        Open-sky budget in seconds.
    """
    # Umbra arc: centered opposite the Sun culmination
    nu = eclipse_half_angle(a, beta_sun)
    if nu == 0.0:
        return 0.0

    umbra_center = uc_sun + math.pi

    # Target-clear arc
    beta_tgt, uc_tgt = target_beta_uc(i, omega, target)
    lam = clear_half_angle(a, beta_tgt, fov_half)
    if lam == 0.0:
        return 0.0

    overlap = arc_intersection(umbra_center, nu, uc_tgt, lam)
    return overlap / mean_motion(a)


def visibility_state(
    u: float,
    umbra_c: float,
    nu: float,
    uc_tgt: float,
    lam_vis: float,
    lam_clear: float,
) -> str:
    """Classify one orbit position into a geometric visibility state.

    Parameters
    ----------
    u : float
        Current argument of latitude (rad).
    umbra_c : float
        Centre of umbra arc = uc_sun + π (rad), the anti-solar direction.
    nu : float
        Umbra half-angle (rad).
    uc_tgt : float
        Target culmination argument of latitude (rad).
    lam_vis : float
        Half-angle (rad) where the target centre clears the Earth limb
        (= ``clear_half_angle(a, beta_tgt, fov_half=0)``).
    lam_clear : float
        Half-angle (rad) where the entire FOV clears the Earth limb
        (= ``clear_half_angle(a, beta_tgt, fov_half)``).

    Returns
    -------
    str
        One of SUNLIT, OCCULTED, PARTIAL, OBSERVABLE.
    """
    in_umbra = nu > 0 and abs(math.remainder(u - umbra_c, 2 * math.pi)) < nu
    if not in_umbra:
        return SUNLIT

    in_vis = lam_vis > 0 and abs(math.remainder(u - uc_tgt, 2 * math.pi)) < lam_vis
    if not in_vis:
        return OCCULTED

    in_clear = lam_clear > 0 and abs(math.remainder(u - uc_tgt, 2 * math.pi)) < lam_clear
    if not in_clear:
        return PARTIAL

    return OBSERVABLE
