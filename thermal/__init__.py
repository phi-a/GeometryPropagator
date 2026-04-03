"""Thermal consumers built on top of geometric view-factor products.

Pipeline
--------
surface_loading_propagate  ->  radiative_background         (W/m² per patch)
                           ->  steady_state_temperature      (K per patch)
                           ->  effective_sink_temperature    (K effective environment)
"""

from .constants import SIGMA_SB                            # noqa: F401
from .background import (SurfaceBackgroundProfile,         # noqa: F401
                         radiative_background)
from .solver import (SurfaceThermalProfile,                # noqa: F401
                     SinkTemperatureProfile,
                     steady_state_temperature,
                     effective_sink_temperature)
