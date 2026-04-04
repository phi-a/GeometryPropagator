"""Thermal consumers built on top of geometric view-factor products.

Pipeline
--------
surface_loading_propagate  ->  radiative_background         (W/m^2 per patch)
                           ->  steady_state_temperature      (K per patch)
                           ->  effective_sink_temperature    (K effective environment)
                           ->  thermal profile plots         (2-D analysis helpers)
"""

from .constants import (                                     # noqa: F401
    SIGMA_SB,
    # Surface coating catalogue  (NA-104-STAR-001-R001, Table 4-3)
    Coating,
    SOLAR_CELL, CLEAR_HARD_ANODISED, WHITE_PAINT_A276_Z93,
    WHITE_SOLDERMASK, SURTEC_650, CLEAR_ANODISED,
    BLACK_HARD_ANODISED, KAPTON_1MIL, GREEN_SOLDERMASK,
    PEEK_COATING, PCL_SBAND_ANTENNA,
    # Bulk material properties
    Material,
    SOLAR_PANEL_DEP_6U,
    # Deployable solar panel wings
    SOLAR_PANEL_ETA_ELECTRICAL,
    SOLAR_PANEL_CELL_ALPHA_SOLAR,
    SOLAR_PANEL_CELL_EPSILON,
    SOLAR_PANEL_BACK_HOT,
    SOLAR_PANEL_BACK_COLD,
    SOLAR_PANEL_BACK_ALPHA_SOLAR,
    SOLAR_PANEL_BACK_EPSILON,
    SOLAR_PANEL_SUBSTRATE_AREAL_CAPACITANCE,
    # Bus radiator plates
    RADIATOR_ALPHA_SOLAR,
    RADIATOR_EPSILON,
)
from .background import (SurfaceBackgroundProfile,         # noqa: F401
                         radiative_background)
from .solver import (SurfaceThermalProfile,                # noqa: F401
                     SinkTemperatureProfile,
                     ShroudProfile,
                     steady_state_temperature,
                     steady_state_temperature_two_sided,
                     transient_temperature,
                     effective_sink_temperature,
                     shroud_temperature)
from .plots import (plot_temperature_trace,                # noqa: F401
                    plot_temperature_heatmap,
                    plot_flux_trace)
from .serialize import save_temperatures, load_temperatures # noqa: F401
