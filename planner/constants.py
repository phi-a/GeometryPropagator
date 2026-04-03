"""Physical and mission constants for the DarkNESS imaging scheduler.

All constants live here. No module should hardcode these values.
Grouped by domain for readability and C/C++ transcription.
"""

import math

# ── WGS-84 / Earth ──────────────────────────────────────────────
MU = 3.986004418e14       # m^3/s^2  gravitational parameter
R_E = 6.3781e6            # m        equatorial radius
J2 = 1.08263e-3           #          second zonal harmonic

# ── Solar / Astronomical ────────────────────────────────────────
R_SUN = 6.957e8           # m        solar radius
AU = 1.496e11             # m        astronomical unit
J2000 = 2451545.0         #          Julian Date of J2000.0 epoch

# Solar ephemeris (Astronomical Almanac, low-precision)
SUN_L0 = 280.460          # deg      mean longitude at J2000
SUN_L_RATE = 0.9856474    # deg/day  mean longitude rate
SUN_G0 = 357.528          # deg      mean anomaly at J2000
SUN_G_RATE = 0.9856003    # deg/day  mean anomaly rate
SUN_EQ1 = 1.915           # deg      equation of center, 1st harmonic
SUN_EQ2 = 0.020           # deg      equation of center, 2nd harmonic
OBLIQUITY = 23.439        # deg      mean obliquity at J2000
OBLIQUITY_RATE = 4e-7     # deg/day  obliquity precession rate

# Earth-Sun distance coefficients (AU units)
DIST_A0 = 1.00014         #          baseline distance / AU
DIST_E1 = 0.01671         #          eccentricity coefficient, 1st
DIST_E2 = 0.00014         #          eccentricity coefficient, 2nd

# ── Time ─────────────────────────────────────────────────────────
SECONDS_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0

# ── TLE format ───────────────────────────────────────────────────
TLE_LINE_LEN = 69
TLE_Y2K_PIVOT = 57        # 2-digit year >= 57 → 1900s, else 2000s

# ── Targets (J2000 equatorial) ───────────────────────────────────
SGR_A_RA_DEG = 266.4168   # deg      Sgr A* right ascension
SGR_A_DEC_DEG = -29.0078  # deg      Sgr A* declination

# ── Default instrument ───────────────────────────────────────────
DEFAULT_STARTUP_S = 60.0  # s        payload boot time
DEFAULT_N_PIXELS = 1_300_000
DEFAULT_READ_RATE = 14_000  # pixels/s
DEFAULT_N_MAX = 20        #          max readout samples
DEFAULT_FOV_HALF_DEG = 10.0  # deg   half-cone FOV
