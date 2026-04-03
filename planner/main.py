"""Example: compute imaging schedule for a LEO X-ray payload."""

import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt

from constants import R_E, DEFAULT_N_MAX, DEFAULT_FOV_HALF_DEG
from optimize import Instrument
from plots import (RunBundle, series, landscape, timeline, heatmap,
                    thermal_series, thermal_landscape)
from plots.anim import arc_diagram_animated, landscape_animated
from schedule import print_schedule, schedule_range
from target import GALACTIC_CENTER

__all__ = []
PLOT = True
THERMAL = False
GIF = True
# --- Example orbit (circular LEO, ~500 km altitude) ---
A = R_E + 500e3        # semi-major axis (m)
E = 0.0                 # circular
I = math.radians(51.6)  # inclination
OMEGA0 = math.radians(45)  # initial RAAN
EPOCH = datetime(2027, 3, 21, tzinfo=timezone.utc)  # vernal equinox
END_EPOCH = datetime(2027, 9, 22, tzinfo=timezone.utc)  # autumnal equinox
FOV_HALF = math.radians(DEFAULT_FOV_HALF_DEG)
RAAN_SWEEP_STEP_DEG = 1
LANDSCAPE_STYLE_KEY = None  # hidden painterly variant; "qianli" still works as an alias

# --- Instrument ---
inst = Instrument()

# --- Resolution (single control) ---
# 24 = daily (fast).  6 = 4x per day.  1.5 = one sample per orbit (~16/day at 500 km).
N_MAX = DEFAULT_N_MAX
SCHEDULE_STEP_HOURS = 1

# --- Schedule for X days ---
start = EPOCH
end = END_EPOCH
results = schedule_range(start, end, A, E, I, OMEGA0, EPOCH, inst,
                         n_max=N_MAX,
                         step_hours=SCHEDULE_STEP_HOURS,
                         target=GALACTIC_CENTER, fov_half=FOV_HALF,
                         thermal=THERMAL)

print(f"Altitude: {(A - R_E)/1e3:.0f} km | Inclination: {math.degrees(I):.1f} deg")
print(f"Window: {EPOCH:%Y-%m-%d %H:%M} UTC -> {END_EPOCH:%Y-%m-%d %H:%M} UTC")
print(f"Readout per sample: {inst.readout_per_sample:.1f} s")
print()
# print_schedule(results, inst)

# --- Canonical run bundle ---
bundle = RunBundle(results=results, a=A, e=E, i=I, omega0=OMEGA0,
                   epoch=EPOCH, inst=inst, target=GALACTIC_CENTER,
                   fov_half=FOV_HALF)

# plots
if PLOT:
    series(bundle)
    landscape(bundle, layer="visibility")
    # circle(bundle)
    # timeline(bundle, layer="visibility")
    # heatmap(bundle, layer="visibility")
    if THERMAL:
        thermal_series(bundle)
        thermal_landscape(bundle, arc="sunlit")
        thermal_landscape(bundle, arc="umbra")
    # arc_diagram(bundle)

    plt.show()

# GIFs
if GIF:
    # equatorial_animated(bundle, duration_s=24.0)
    # circle_animated(bundle, duration_s=24.0, raan_step=10)
    landscape_animated(bundle, step_hours=SCHEDULE_STEP_HOURS,
                       duration_s=24.0, raan_step=RAAN_SWEEP_STEP_DEG,
                       style_key=LANDSCAPE_STYLE_KEY)
    # heatmap_animated(bundle, step_hours=SCHEDULE_STEP_HOURS,
    #                  duration_s=24.0, raan_step=1)
    # arc_diagram_animated(bundle, duration_s=60.0)
