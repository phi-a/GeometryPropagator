"""CubeSat geometry builders and reusable body-fixed surface models."""

from .surfaces import (CubeSatGeometry, RealizedGeometry, RectSurface,   # noqa: F401
                       SurfaceNode, flip_surface, mount, rect_patch_grid)
from .builder import build_6u_double_deployable
from .inspect import (surface_by_normal, face_frame_labels,          # noqa: F401
                      signed_axis_label, opposite_axis_label,
                      surface_body_role,
                      print_surface_summary, print_mounted_role_table)
from .scene3d import scene, animate                                   # noqa: F401
