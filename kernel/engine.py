"""Feasible-attitude engine.

Pure S2 geometry. No orbit math. No backend imports.
Takes unit vectors, returns Region + State.

Usage:
    caps = build(s_hat, t_hat, e_hat, THETA_F, ALPHA_EXCL, ALPHA_EARTH)
    rgn  = Region(caps=caps, mesh=vec.sphere(n)).solve()
    st   = state(rgn)
"""

from dataclasses import dataclass
import numpy as np
from . import vec, constraint
from .region import Region


@dataclass
class State:
    """Instantaneous S2 evaluation."""
    feasible: bool
    area: float          # feasible solid angle [sr]
    fraction: float      # feasible fraction [0, 1]
    margin: float        # max min-margin [rad]
    center: np.ndarray   # feasible centroid
    safest: np.ndarray   # max-margin direction
    dominant: str        # globally tightest constraint label
    dominance: dict      # label -> fraction of sphere dominated


def build(s_hat, t_hat, e_hat, THETA_F, ALPHA_EXCL, ALPHA_EARTH):
    """Build constraint caps from direction vectors.

    Parameters
    ----------
    s_hat : (3,) Sun unit vector.
    t_hat : (3,) Target unit vector.
    e_hat : (3,) Earth/nadir proxy unit vector.
    THETA_F : float  FOV half-angle [rad].
    ALPHA_EXCL : float  Sun exclusion half-angle [rad].
    ALPHA_EARTH : float  Earth keepout half-angle [rad].

    Returns
    -------
    list[Cap]
    """
    return [
        constraint.fov(t_hat, THETA_F),
        constraint.sun(s_hat, ALPHA_EXCL),
        constraint.earth(e_hat, ALPHA_EARTH),
    ]


def state(rgn):
    """Extract State from a solved Region.

    Parameters
    ----------
    rgn : Region (must be .solve()'d)

    Returns
    -------
    State
    """
    safest, marg = rgn.maxmargin()
    return State(
        feasible=rgn.exists(),
        area=rgn.area(),
        fraction=rgn.fraction(),
        margin=marg if marg > float('-inf') else 0.0,
        center=rgn.center(),
        safest=safest,
        dominant=rgn.bottleneck(),
        dominance=rgn.dominance(),
    )
