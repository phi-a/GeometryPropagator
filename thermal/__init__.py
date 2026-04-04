"""Thermal consumers built on top of geometric view-factor products.

Pipeline
--------
surface_loading_propagate  ->  radiative_background         (W/m^2 per patch)
                           ->  steady_state_temperature      (K per patch)
                           ->  effective_sink_temperature    (K effective environment)
                           ->  thermal profile plots         (2-D analysis helpers)
"""

from .constants import (SIGMA_SB,                           # noqa: F401
                        SOLAR_PANEL_CELL_ALPHA_SOLAR,
                        SOLAR_PANEL_CELL_EPSILON,
                        SOLAR_PANEL_BACK_ALPHA_SOLAR,
                        SOLAR_PANEL_BACK_EPSILON,
                        SOLAR_PANEL_SUBSTRATE_AREAL_CAPACITANCE)
from .background import (SurfaceBackgroundProfile,         # noqa: F401
                         radiative_background)
from .solver import (SurfaceThermalProfile,                # noqa: F401
                     SinkTemperatureProfile,
                     steady_state_temperature,
                     steady_state_temperature_two_sided,
                     transient_temperature,
                     effective_sink_temperature)
from .plots import (plot_temperature_trace,                # noqa: F401
                    plot_temperature_heatmap)
