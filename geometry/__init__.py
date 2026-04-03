"""Spacecraft geometry kernel: orbit mechanics, attitude, and body shape.

Active modules
--------------
    orbit        orbit configuration and precomputed geometry
    so3          SO(3) rotation tools
    laws         steady-state attitude laws
    transitions  finite-rate wrappers around base laws
    CubeSat      body-fixed CubeSat geometry builders and realizations

Legacy modules
--------------
    legacy       older scalar flat-plate approximations kept for reference

View-factor layer
-----------------
Directional visibility, Earth-disk quadrature, occlusion, and orbit-sweep
propagators now live in the ``viewfactor`` package:

    from viewfactor import (earth_loading_propagate, panel_loading_propagate,
                            spacecraft_occlusion_mask, integrate_surface_response,
                            EarthDiskQuadrature, RectangularPanel)
"""

from importlib import import_module

from .orbit import Orbit, beta_uc, direction                           # noqa: F401
from .so3 import SO3                                                   # noqa: F401
from .laws import (LVLHFixed, TargetTracking, TargetTrackingNadirRoll,  # noqa: F401
                   SunTracking, InertialDrift, ModeSwitch)              # noqa: F401
from .transitions import SlewModeSwitch                                # noqa: F401
from .constants import FACES, S0, J_IR, A_ALB                          # noqa: F401
from .CubeSat import (CubeSatGeometry, RealizedGeometry, RectSurface,  # noqa: F401
                      SurfaceNode, build_6u_double_deployable, mount)
from .legacy import (earth_vf, propagate, thermal_propagate,           # noqa: F401
                     ViewFactorProfile, ThermalProfile)
from . import legacy                                                   # noqa: F401

_VIEWFACTOR_EXPORTS = {
    'EarthDiskQuadrature',
    'EarthDiskSamples',
    'FACE_LOCAL_FRAMES',
    'AzimuthElevationMask',
    'face_coordinates',
    'integrate_face_response',
    'RectangularPanel',
    'PanelLoadingProfile',
    'spacecraft_occlusion_mask',
    'integrate_surface_response',
    'hemisphere_group_view',
    'earth_loading_propagate',
    'EarthLoadingProfile',
    'panel_loading_propagate',
    'surface_loading_propagate',
    'SurfaceLoadingProfile',
}


def __getattr__(name):
    """Lazy backward-compat access to names that moved into viewfactor."""
    if name in _VIEWFACTOR_EXPORTS:
        module = import_module('viewfactor')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
