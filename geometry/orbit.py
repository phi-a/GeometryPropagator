"""Orbit geometry configuration.

Single source of truth: orbital elements + epoch -> all derived geometry.
Self-contained astronomy (Sun ephemeris, beta angles, eclipse model).
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# -- Physical constants --------------------------------------------------
R_E   = 6.3781e6          # m   Earth equatorial radius
MU    = 3.986004418e14    # m^3/s^2  gravitational parameter
R_SUN = 6.957e8           # m   solar radius
AU    = 1.496e11          # m   astronomical unit
J2000 = 2451545.0         #     Julian Date of J2000.0

# Solar ephemeris coefficients (Astronomical Almanac, low-precision)
_L0, _LR     = 280.460, 0.9856474     # mean longitude  deg, deg/day
_G0, _GR     = 357.528, 0.9856003     # mean anomaly    deg, deg/day
_EQ1, _EQ2   = 1.915, 0.020           # equation of centre  deg
_EPS0, _EPSR = 23.439, 4e-7           # obliquity  deg, deg/day

# -- LVLH convention -----------------------------------------------------
#   index 0 = along-track  (T, velocity)
#   index 1 = cross-track  (W, orbit normal)
#   index 2 = radial       (R, zenith)
NADIR = np.array([0.0, 0.0, -1.0])


# -- Astronomy helpers (private) -----------------------------------------

def _jd(dt):
    """UTC datetime -> Julian Date."""
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = (dt.day + (153 * m + 2) // 5
           + 365 * y + y // 4 - y // 100 + y // 400 - 32045)
    return jdn + (dt.hour - 12 + dt.minute / 60.0 + dt.second / 3600.0) / 24.0


def _sun_ra_dec(dt):
    """Sun RA and Dec [rad] at UTC datetime."""
    n = _jd(dt) - J2000
    g = math.radians((_G0 + _GR * n) % 360)
    lam = math.radians(
        (_L0 + _LR * n + _EQ1 * math.sin(g) + _EQ2 * math.sin(2 * g)) % 360
    )
    eps = math.radians(_EPS0 - _EPSR * n)
    ra  = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    return ra, dec


def beta_uc(i, omega, ra, dec):
    """Beta angle and culmination argument of latitude for any inertial direction.

    Parameters
    ----------
    i     : float  orbit inclination [rad]
    omega : float  RAAN [rad]
    ra    : float  right ascension of direction [rad]
    dec   : float  declination of direction [rad]

    Returns
    -------
    beta : float  beta angle [rad]
    uc   : float  culmination argument of latitude [rad]
    """
    sd, cd = math.sin(dec), math.cos(dec)
    si, ci = math.sin(i),   math.cos(i)
    sin_beta = si * cd * math.sin(omega - ra) + ci * sd
    beta = math.asin(max(-1.0, min(1.0, sin_beta)))
    gx = cd * math.cos(ra - omega)
    gy = ci * cd * math.sin(ra - omega) + si * sd
    uc = math.atan2(gy, gx)
    return beta, uc


def _eclipse_half(a, beta):
    """Eclipse half-angle [rad], conical umbra model.  0 if no eclipse."""
    k   = R_E / a
    eps = (R_SUN - R_E) / AU
    cos_beta = abs(math.cos(beta))
    if cos_beta < 1e-15:
        return 0.0
    arg = (k * eps + math.sqrt((1 - k**2) * (1 - eps**2))) / cos_beta
    return 0.0 if arg >= 1.0 else math.acos(arg)


# -- Direction in LVLH ---------------------------------------------------

def direction(beta, uc, u):
    """Unit vector in LVLH (T, W, R) at argument of latitude *u*.

    Parameters
    ----------
    beta : float  beta angle of the direction [rad]
    uc   : float  argument of latitude of culmination [rad]
    u    : float  spacecraft argument of latitude [rad]
    """
    cb = math.cos(beta)
    du = u - uc
    return np.array([-cb * math.sin(du), math.sin(beta), cb * math.cos(du)])


# -- Orbit dataclass ------------------------------------------------------

@dataclass(frozen=True)
class Orbit:
    """Immutable orbit geometry.  Single configuration point.

    Construct via ``Orbit.from_epoch(...)`` for automatic Sun/target
    geometry, or directly for precomputed values.
    """
    a:        float          # semi-major axis [m]
    i:        float          # inclination [rad]
    omega:    float          # RAAN [rad]
    H:        float          # normalised altitude  a / R_E
    rho:      float          # Earth angular radius from S/C [rad]
    n:        float          # mean motion [rad/s]
    beta_sun: float          # Sun beta angle [rad]
    uc_sun:   float          # Sun culmination arg-of-lat [rad]
    nu:       float          # eclipse half-angle [rad]
    beta_tgt: float = None   # target beta angle [rad]
    uc_tgt:   float = None   # target culmination [rad]
    epoch:    object = None  # reference epoch (informational)

    # -- factory -----------------------------------------------------------

    @staticmethod
    def from_epoch(a, i, omega, epoch, target_radec=None):
        """Construct from orbital elements and UTC epoch.

        Parameters
        ----------
        a     : float     semi-major axis [m]
        i     : float     inclination [rad]
        omega : float     RAAN [rad]
        epoch : datetime  UTC epoch for Sun ephemeris
        target_radec : tuple (ra_rad, dec_rad), optional
        """
        ra_s, dec_s = _sun_ra_dec(epoch)
        bs, us = beta_uc(i, omega, ra_s, dec_s)
        nu = _eclipse_half(a, bs)

        bt, ut = None, None
        if target_radec is not None:
            bt, ut = beta_uc(i, omega, *target_radec)

        return Orbit(
            a=a, i=i, omega=omega,
            H=a / R_E,
            rho=math.asin(R_E / a),
            n=math.sqrt(MU / a**3),
            beta_sun=bs, uc_sun=us, nu=nu,
            beta_tgt=bt, uc_tgt=ut,
            epoch=epoch,
        )

    # -- direction queries -------------------------------------------------

    def sun_dir(self, u):
        """Sun direction in LVLH at argument of latitude *u*."""
        return direction(self.beta_sun, self.uc_sun, u)

    def target_dir(self, u):
        """Target direction in LVLH at argument of latitude *u*."""
        return direction(self.beta_tgt, self.uc_tgt, u)

    def in_eclipse(self, u):
        """True if *u* falls within the eclipse (umbra) arc."""
        if self.nu <= 0:
            return False
        return abs(math.remainder(u - self.uc_sun - math.pi, 2 * math.pi)) < self.nu

    @property
    def period(self):
        """Orbital period [s]."""
        return 2 * math.pi / self.n

    # -- ECI vector queries ------------------------------------------------

    def r_hat_eci(self, u):
        """Radial (zenith) unit vector in ECI at argument of latitude *u*."""
        cu, su = math.cos(u), math.sin(u)
        cO, sO = math.cos(self.omega), math.sin(self.omega)
        ci, si = math.cos(self.i),     math.sin(self.i)
        return np.array([
            cu * cO - su * ci * sO,
            cu * sO + su * ci * cO,
            su * si,
        ])

    def nadir_eci(self, u):
        """Nadir (toward Earth centre) unit vector in ECI."""
        return -self.r_hat_eci(u)

    def v_hat_eci(self, u):
        """Along-track (velocity) unit vector in ECI at argument *u*."""
        cu, su = math.cos(u), math.sin(u)
        cO, sO = math.cos(self.omega), math.sin(self.omega)
        ci, si = math.cos(self.i),     math.sin(self.i)
        return np.array([
            -su * cO - cu * ci * sO,
            -su * sO + cu * ci * cO,
             cu * si,
        ])

    @property
    def h_hat_eci(self):
        """Orbit-normal (angular momentum) unit vector in ECI."""
        cO, sO = math.cos(self.omega), math.sin(self.omega)
        si, ci = math.sin(self.i),     math.cos(self.i)
        return np.array([si * sO, -si * cO, ci])

    def sun_eci(self):
        """Sun direction unit vector in ECI (constant over one orbit)."""
        if self.epoch is None:
            raise ValueError("Orbit has no epoch — cannot compute sun_eci()")
        ra, dec = _sun_ra_dec(self.epoch)
        cd = math.cos(dec)
        return np.array([math.cos(ra) * cd, math.sin(ra) * cd, math.sin(dec)])

    def eci_from_lvlh(self, u):
        """3×3 rotation matrix mapping LVLH vectors to ECI at argument *u*.

        Columns: [v_hat, h_hat, r_hat] — the LVLH (T, W, R) axes in ECI.
        """
        return np.column_stack([self.v_hat_eci(u), self.h_hat_eci, self.r_hat_eci(u)])
