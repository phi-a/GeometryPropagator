"""View-factor layer: geometric visibility between spacecraft surfaces and sources.

This package computes what each surface patch can see, expressed as
dimensionless geometric factors. It does not compute watts or temperatures.

Modules
-------
earthdisk   Earth-disk quadrature, face-coordinate transforms, directional masks
panel       Patch-resolved rectangular radiator geometry
occlusion   Spacecraft self-occlusion ray tests and group-view integration
propagator  Orbit-sweep propagators returning EarthLoadingProfile / PanelLoadingProfile

Typical usage
-------------
    from geometry import Orbit, LVLHFixed, build_6u_double_deployable
    from viewfactor import earth_loading_propagate, spacecraft_occlusion_mask

    orbit    = Orbit.from_epoch(...)
    law      = LVLHFixed()
    realized = build_6u_double_deployable().realize()

    profile  = earth_loading_propagate(orbit, law)
    mask     = spacecraft_occlusion_mask(realized, 'bus_+Y', dirs_body)
"""

from .earthdisk import (EarthDiskQuadrature, EarthDiskSamples,       # noqa: F401
                        FACE_LOCAL_FRAMES, AzimuthElevationMask,
                        face_coordinates, integrate_face_response)
from .panel import RectangularPanel, PanelLoadingProfile              # noqa: F401
from .occlusion import (spacecraft_occlusion_mask,                   # noqa: F401
                        integrate_surface_response,
                        hemisphere_group_view)
from .propagator import (earth_loading_propagate, EarthLoadingProfile,  # noqa: F401
                         panel_loading_propagate,
                         surface_loading_propagate, SurfaceLoadingProfile)
from .sampling import hemisphere_directions                           # noqa: F401
