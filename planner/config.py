"""Central configuration: visibility and operational state definitions, figure defaults.

Figure defaults and two independent classification layers
--------------------------------------
Visibility layer  — geometric obstruction states (arc membership only)
Operational layer — instrument mode states (within the observable window)

Color map dictionaries use integer codes as keys so they compose directly
into a matplotlib ListedColormap via ``list(PHASE_COLORS.values())``.

Codes 0–3: visibility layer
Codes 4–6: operational layer (overwrite observable region when a solution exists)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FIG_FONT = 12
FIG_TITLE = 14

# ---------------------------------------------------------------------------
# Visibility layer  (geometric obstruction)
# ---------------------------------------------------------------------------
#   SUNLIT     — satellite outside the umbra arc; no eclipse
#   OCCULTED   — in eclipse; target centre behind the Earth limb  (deep blue)
#   PARTIAL    — in eclipse; target visible but FOV clips the limb (mid-blue)
#   OBSERVABLE — in eclipse; entire FOV clears the limb           (deep iris)

VISIBILITY_COLORS = {
    0: "#ffffff",   # sunlit    — white
    1: "#1a3070",   # occulted  — deep blue  (full obstruction)
    2: "#5b85c8",   # partial   — mid-blue   (FOV clips Earth limb)
    3: "#3d2b8c",   # observable— deep iris  (full clearance)
}

VISIBILITY_LABELS = {
    0: "Sunlit",
    1: "Occulted",
    2: "Partial",
    3: "Observable",
}

# ---------------------------------------------------------------------------
# Operational layer  (instrument modes, within observable only)
# ---------------------------------------------------------------------------
#   STARTUP  — instrument powering on        (olive)
#   EXPOSURE — science integration           (near-white, high contrast on iris)
#   READOUT  — data readout after exposure   (gray)

OPERATIONAL_COLORS = {
    4: "#6b8e23",   # startup  — olive
    5: "#f0f0f0",   # exposure — near-white
    6: "#808080",   # readout  — gray
}

OPERATIONAL_LABELS = {
    4: "Startup",
    5: "Exposure",
    6: "Readout",
}

# ---------------------------------------------------------------------------
# Combined phase colormap  (visibility base + operational overlay)
# ---------------------------------------------------------------------------
# Ordered list of hex codes indexed 0–6.  Build a ListedColormap with:
#   cmap = ListedColormap(list(PHASE_COLORS.values()))
#   norm = BoundaryNorm(range(len(PHASE_COLORS) + 1), cmap.N)

# ---------------------------------------------------------------------------
# Reference-line colors  (shared across views)
# ---------------------------------------------------------------------------
ECLIPSE_LINE  = "#000000"
OPEN_SKY_LINE = "#2ca02c"

# ---------------------------------------------------------------------------
# Landscape fill colors  (keyed by semantic name per layer)
# ---------------------------------------------------------------------------
LANDSCAPE_VISIBILITY = {
    "observable": VISIBILITY_COLORS[3],   # deep iris
    "partial":    VISIBILITY_COLORS[2],   # mid-blue  (target visible, FOV clips)
    "occulted":   VISIBILITY_COLORS[1],   # deep blue
}

LANDSCAPE_OPERATIONAL = {
    "startup":  OPERATIONAL_COLORS[4],   # olive
    "exposure": OPERATIONAL_COLORS[5],   # near-white
    "readout":  OPERATIONAL_COLORS[6],   # gray
    "unused":   VISIBILITY_COLORS[3],    # deep iris (observable but unscheduled)
    "blocked":  VISIBILITY_COLORS[1],    # deep blue (occulted/blocked)
}

# ---------------------------------------------------------------------------
# Combined phase colormap  (visibility base + operational overlay)
# ---------------------------------------------------------------------------
PHASE_COLORS = {**VISIBILITY_COLORS, **OPERATIONAL_COLORS}
PHASE_LABELS = {**VISIBILITY_LABELS, **OPERATIONAL_LABELS}


def phase_colormap(layer="combined"):
    """Return (ListedColormap, BoundaryNorm) for a phase grid.

    Parameters
    ----------
    layer : str
        ``"visibility"``  — codes 0–3 only (4 colours).
        ``"operational"`` — codes 0,3–6 (observable background + 3 phases).
        ``"combined"``    — codes 0–6 (all 7 colours).
    """
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if layer == "visibility":
        colors = [VISIBILITY_COLORS[k] for k in sorted(VISIBILITY_COLORS)]
    elif layer == "operational":
        # 0=sunlit(white), 1=observable(iris bg), 2=startup, 3=exposure, 4=readout
        colors = [
            VISIBILITY_COLORS[0],    # sunlit  — white
            VISIBILITY_COLORS[3],    # observable background — deep iris
            OPERATIONAL_COLORS[4],   # startup  — olive
            OPERATIONAL_COLORS[5],   # exposure — near-white
            OPERATIONAL_COLORS[6],   # readout  — gray
        ]
    else:
        colors = [PHASE_COLORS[k] for k in sorted(PHASE_COLORS)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
    return cmap, norm


def visibility_legend_handles():
    """Return matplotlib Patch list for the visibility layer legend."""
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color=VISIBILITY_COLORS[k], label=VISIBILITY_LABELS[k])
        for k in sorted(VISIBILITY_COLORS)
        if k > 0   # skip sunlit — implicit background
    ]


def operational_legend_handles():
    """Return matplotlib Patch list for the operational layer legend."""
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color=OPERATIONAL_COLORS[k], label=OPERATIONAL_LABELS[k])
        for k in sorted(OPERATIONAL_COLORS)
    ]
