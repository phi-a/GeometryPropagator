"""SO3 attitude geometry package.

Active layers:
    orbit        orbit configuration and precomputed geometry
    so3          SO(3) rotation tools
    laws         steady-state attitude laws
    transitions  finite-rate wrappers around base laws
    CubeSat      body-fixed CubeSat geometry builders and realizations
    earthdisk    Earth-disk quadrature and face coordinates
    panel        panel-resolved radiator geometry
    propagator   active disk-integrated and panel-resolved sweeps

Legacy layers:
    legacy       older scalar flat-plate approximations kept for reference
"""

from .orbit import Orbit, beta_uc, direction                           # noqa: F401
from .so3 import SO3                                                   # noqa: F401
from .laws import (LVLHFixed, TargetTracking, TargetTrackingNadirRoll,  # noqa: F401
                   SunTracking, InertialDrift, ModeSwitch)              # noqa: F401
from .transitions import SlewModeSwitch                                # noqa: F401
from .constants import FACES, S0, J_IR, A_ALB                          # noqa: F401
from .CubeSat import (CubeSatGeometry, RealizedGeometry, RectSurface,  # noqa: F401
                      SurfaceNode, build_6u_double_deployable)
from .earthdisk import (EarthDiskQuadrature, EarthDiskSamples,         # noqa: F401
                        FACE_LOCAL_FRAMES, AzimuthElevationMask)        # noqa: F401
from .panel import RectangularPanel, PanelLoadingProfile               # noqa: F401
from .occlusion import (spacecraft_occlusion_mask,                    # noqa: F401
                        integrate_surface_response)
from .propagator import (earth_loading_propagate, EarthLoadingProfile,  # noqa: F401
                         panel_loading_propagate)
from .legacy import (earth_vf, propagate, thermal_propagate,           # noqa: F401
                     ViewFactorProfile, ThermalProfile)
from . import legacy                                                   # noqa: F401
