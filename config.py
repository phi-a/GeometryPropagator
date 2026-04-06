"""Central path configuration for all notebooks and scripts.

Update paths here — no need to touch individual notebooks.
"""
from pathlib import Path

OUTPUTS     = Path('outputs')
GEO_DIR     = OUTPUTS / 'geometry'
VF_DIR      = OUTPUTS / 'viewfactor'
THERMAL_DIR = OUTPUTS / 'thermal'
PLOTS_DIR   = OUTPUTS / 'plots'

# ── Geometry ───────────────────────────────────────────────────────────────
SPACECRAFT_JSON = GEO_DIR / 'cubesat_2416.json'
BODY_ROLES_JSON = GEO_DIR / 'darkness_map.json'

# ── View factors ───────────────────────────────────────────────────────────
VF_NPZ       = VF_DIR / 'ISSVF45_2416_360.npz'
VF_META_JSON = VF_DIR / 'ISSVF45_2416_360_meta.json'

# ── Thermal ────────────────────────────────────────────────────────────────
PANEL_TEMP_NPZ = THERMAL_DIR / 'ISST45_2416_360.npz'   # run_solarpanel → run_background
