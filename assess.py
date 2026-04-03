"""Assess feasible attitude set A(t) for DarkNESS.

Driver: optimizeSNR provides orbit/sun/target geometry.
Kernel: pure S2 intersection of constraint caps.

Run from DAO root:
    python assess.py
"""

import math
import sys
import numpy as np
from datetime import datetime, timezone, timedelta

# backend
sys.path.insert(0, "planner")
from sun import sun_ra_dec                                       # noqa: E402
from orbit import (j2_raan_rate, propagate_raan, sun_beta_uc,    # noqa: E402
                   mean_motion, eclipse_half_angle)
from target import (GALACTIC_CENTER, target_beta_uc,             # noqa: E402
                    clear_half_angle, arc_intersection)
from sun import sun_dist                                         # noqa: E402

# kernel
from kernel import vec                                           # noqa: E402
from kernel.engine import build, state                           # noqa: E402
from kernel.region import Region                                 # noqa: E402

# -- mission ------------------------------------------------------------------

A       = 6.3781e6 + 500e3
E       = 0.0
I       = math.radians(51.6)
OMEGA_0 = math.radians(180.0)
EPOCH   = datetime(2027, 3, 21, tzinfo=timezone.utc)
TARGET  = GALACTIC_CENTER
THETA_F = math.radians(10.0)
ALPHA_EXCL  = math.radians(45.0)
ALPHA_EARTH = math.radians(70.0)
N_MESH      = 4000

SIGMA = {
    "FOV":        math.radians(2.0),
    "Sun Excl":   math.radians(1.0),
    "Earth Limb": math.radians(3.0),
}


def directions(dt, omega):
    """Get unit vectors from optimizeSNR backend."""
    ra_sun, dec_sun = sun_ra_dec(dt)
    s_hat = vec.radec(ra_sun, dec_sun)
    t_hat = vec.radec(TARGET.ra_rad, TARGET.dec_rad)
    e_hat = np.array([
        -math.sin(I) * math.sin(omega),
         math.sin(I) * math.cos(omega),
        -math.cos(I),
    ])
    return s_hat, t_hat, e_hat


def arc_open(dt, omega):
    """Arc-method open-sky time [s] from optimizeSNR."""
    beta_sun, uc_sun = sun_beta_uc(I, omega, dt)
    beta_tgt, uc_tgt = target_beta_uc(I, omega, TARGET)
    D = sun_dist(dt)
    nu = eclipse_half_angle(A, beta_sun, D)
    lam = clear_half_angle(A, beta_tgt, THETA_F)
    n_mm = mean_motion(A)
    overlap = arc_intersection(uc_sun + math.pi, nu, uc_tgt, lam)
    return overlap / n_mm if n_mm > 0 else 0.0


# -- single instant ------------------------------------------------------------

print("-" * 60)
print("Single-instant evaluation")
print("-" * 60)

omega_dot = j2_raan_rate(A, E, I)
omega = propagate_raan(OMEGA_0, omega_dot, 0.0)
dt = EPOCH

s, t, e = directions(dt, omega)
caps = build(s, t, e, THETA_F, ALPHA_EXCL, ALPHA_EARTH)
mesh = vec.sphere(N_MESH)
rgn = Region(caps=caps, mesh=mesh).solve()
st = state(rgn)
T_open = arc_open(dt, omega)

print(f"  feasible    : {st.feasible}")
print(f"  area        : {st.area:.3f} sr  ({st.fraction*100:.1f}%)")
print(f"  margin      : {np.degrees(st.margin):.1f} deg")
print(f"  T_open      : {T_open:.1f} s  (arc method)")
print(f"  bottleneck  : {st.dominant}")
for k, v in st.dominance.items():
    print(f"    {k:>12s} : {v*100:.0f}%")

# -- uncertainty ---------------------------------------------------------------

print()
print("-" * 60)
print("Uncertainty-widened feasible set")
print("-" * 60)

rgn_u = rgn.uncertain(SIGMA)
st_u = state(rgn_u)

print(f"  deterministic area   : {rgn.area():.4f} sr")
print(f"  uncertain area       : {rgn_u.area():.4f} sr")
print(f"  area shrinkage       : {(1 - rgn_u.area()/max(rgn.area(), 1e-12))*100:.1f}%")
print(f"  deterministic margin : {np.degrees(st.margin):.1f} deg")
print(f"  uncertain margin     : {np.degrees(st_u.margin):.1f} deg")
print(f"  bottleneck (uncert)  : {st_u.dominant}")

# -- dominance -----------------------------------------------------------------

print()
print("-" * 60)
print("Constraint dominance")
print("-" * 60)

for label, frac in sorted(st.dominance.items(), key=lambda x: -x[1]):
    bar = "#" * int(frac * 50)
    print(f"  {label:>12s} : {frac*100:5.1f}%  {bar}")

# -- year sweep ----------------------------------------------------------------

print()
print("-" * 60)
print("Year sweep  (1-hour, 8760 evaluations)")
print("-" * 60)

mesh2k = vec.sphere(2000)
results = []
for h in range(365 * 24):
    dt_h = EPOCH + timedelta(hours=h)
    omega_h = propagate_raan(OMEGA_0, omega_dot, h * 3600.0)
    s, t, e = directions(dt_h, omega_h)
    caps = build(s, t, e, THETA_F, ALPHA_EXCL, ALPHA_EARTH)
    rgn_h = Region(caps=caps, mesh=mesh2k).solve()
    results.append(state(rgn_h))
    if h % 1000 == 0 and h > 0:
        print(f"  ... {h}/8760")

feasible_n = sum(r.feasible for r in results)
areas   = [r.area for r in results]
margins = [r.margin for r in results]

print(f"  feasible hours : {feasible_n} / {len(results)}")
print(f"  mean area      : {np.mean(areas):.3f} sr")
print(f"  mean margin    : {np.degrees(np.mean(margins)):.1f} deg")

from collections import Counter
bots = Counter(r.dominant for r in results)
print(f"\n  Yearly bottleneck:")
for label, count in bots.most_common():
    print(f"    {label:>12s} : {count} hrs ({count/len(results)*100:.0f}%)")
