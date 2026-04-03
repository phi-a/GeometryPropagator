"""Constraint builders: mission geometry -> labeled Caps on S².

Each function returns a Cap. The label is used for the
constraint-dominance map.

Labels use operational language:
    FOV         - target must be inside field of view
    Sun Excl    - boresight must avoid Sun
    Earth Limb  - FOV must clear Earth limb

Backend: directions come from optimizeSNR schedule_date results.
"""

import numpy as np
from . import vec
from .cap import Cap


def fov(t_hat, ALPHA_FOV):
    """FOV clearance: boresight must contain target."""
    return Cap(c=vec.hat(t_hat), ALPHA=ALPHA_FOV,
               sense="inside", label="FOV")


def sun(s_hat, ALPHA_EXCL):
    """Sun exclusion: boresight must avoid Sun."""
    return Cap(c=vec.hat(s_hat), ALPHA=ALPHA_EXCL,
               sense="outside", label="Sun Excl")


def earth(e_hat, ALPHA_KEEP):
    """Earth limb: FOV must clear Earth."""
    return Cap(c=vec.hat(e_hat), ALPHA=ALPHA_KEEP,
               sense="outside", label="Earth Limb")
