"""Orbit propagation and eclipse geometry."""

import math
from datetime import datetime, timedelta

from constants import MU, R_E, J2, R_SUN, AU
from sun import sun_ra_dec


def mean_motion(a: float) -> float:
    """Mean motion n (rad/s) for semi-major axis a (m)."""
    return math.sqrt(MU / a**3)


def j2_raan_rate(a: float, e: float, i: float) -> float:
    """Secular J2 RAAN drift rate (rad/s)."""
    n = mean_motion(a)
    return -1.5 * J2 * n * (R_E / a) ** 2 * math.cos(i) / (1 - e**2) ** 2


def propagate_raan(omega0: float, omega_dot: float, dt_seconds: float) -> float:
    """Propagate RAAN (rad) over dt_seconds."""
    return omega0 + omega_dot * dt_seconds


def raan_for_ltan(dt: datetime, ltan_hours: float) -> float:
    """Return the RAAN (rad) that yields the requested LTAN at *dt*.

    LTAN is referenced to local solar noon, so 12.0 h means the ascending
    node is aligned with the Sun right ascension at the given epoch.
    """
    ra_sun, _ = sun_ra_dec(dt)
    return (ra_sun + math.radians(15.0 * (ltan_hours - 12.0))) % (2 * math.pi)


def ltan_for_raan(dt: datetime, omega: float) -> float:
    """Return the LTAN (hours) corresponding to RAAN *omega* at *dt*."""
    ra_sun, _ = sun_ra_dec(dt)
    delta_deg = math.degrees((omega - ra_sun) % (2 * math.pi))
    return (12.0 + delta_deg / 15.0) % 24.0


def beta_angle(i: float, omega: float, dt: datetime) -> float:
    """Compute Sun beta angle (rad) for orbit plane normal vs Sun direction."""
    ra_sun, dec_sun = sun_ra_dec(dt)
    sin_beta = (
        math.sin(i) * math.cos(dec_sun) * math.sin(omega - ra_sun)
        + math.cos(i) * math.sin(dec_sun)
    )
    return math.asin(max(-1.0, min(1.0, sin_beta)))


def eclipse_half_angle(a: float, beta: float, D: float = AU) -> float:
    """Eclipse half-angle (rad), conical umbra model. Returns 0 if no eclipse."""
    k, eps = R_E / a, (R_SUN - R_E) / D
    arg = (k * eps + math.sqrt((1 - k**2) * (1 - eps**2))) / abs(math.cos(beta))
    return 0.0 if arg >= 1.0 else math.acos(arg)


def eclipse_duration(a: float, beta: float, D: float = AU) -> float:
    """Eclipse duration (seconds) for circular orbit. Returns 0 if no eclipse."""
    nu = eclipse_half_angle(a, beta, D)
    if nu == 0.0:
        return 0.0
    return 2.0 * nu / mean_motion(a)


def sun_beta_uc(i: float, omega: float, dt: datetime) -> tuple[float, float]:
    """Sun beta angle and argument-of-latitude of culmination.

    Parameters
    ----------
    i : float
        Inclination (rad).
    omega : float
        RAAN (rad).
    dt : datetime
        UTC datetime for Sun position.

    Returns
    -------
    beta : float
        Sun beta angle (rad).
    uc : float
        Argument of latitude where Sun culminates (rad).
    """
    ra_sun, dec_sun = sun_ra_dec(dt)

    sin_beta = (
        math.sin(i) * math.cos(dec_sun) * math.sin(omega - ra_sun)
        + math.cos(i) * math.sin(dec_sun)
    )
    beta = math.asin(max(-1.0, min(1.0, sin_beta)))

    g_x = math.cos(dec_sun) * math.cos(ra_sun - omega)
    g_y = (math.cos(i) * math.cos(dec_sun) * math.sin(ra_sun - omega)
           + math.sin(i) * math.sin(dec_sun))
    uc = math.atan2(g_y, g_x)

    return beta, uc
