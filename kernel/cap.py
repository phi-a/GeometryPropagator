"""Spherical cap primitive on S².

A Cap is a closed region {b in S² : angle(b, center) <= ALPHA}.
Sense controls inclusion ('inside') or exclusion ('outside').

Uncertainty: Cap.widen(SIGMA) returns a new Cap whose half-angle
is inflated by angular uncertainty SIGMA, implementing the
chance-constraint P(g_i >= 0) >= p_i as geometric margin.
"""

from dataclasses import dataclass
import numpy as np
from . import vec


@dataclass(frozen=True, slots=True)
class Cap:
    """Spherical cap or its complement.

    Parameters
    ----------
    c : ndarray, shape (3,)
        Unit-vector center.
    ALPHA : float
        Half-angle [rad].
    sense : str
        'inside' = cap interior, 'outside' = cap complement.
    label : str
        Constraint name for dominance map.
    """
    c: np.ndarray
    ALPHA: float
    sense: str = "inside"
    label: str = ""

    def contains(self, b):
        """Membership test. b: (3,) or (n,3) -> bool or (n,)."""
        d = vec.dot(b, self.c)
        inside = d >= np.cos(self.ALPHA)
        return inside if self.sense == "inside" else ~inside

    def margin(self, b):
        """Signed geodesic margin to boundary [rad].

        Positive = inside feasible side.
        b: (3,) or (n,3) -> float or (n,).
        """
        ang = vec.angle(b, self.c)
        raw = self.ALPHA - ang
        return raw if self.sense == "inside" else -raw

    def widen(self, SIGMA):
        """Return uncertainty-inflated Cap.

        For 'inside' caps: shrink by SIGMA (conservative).
        For 'outside' caps: grow exclusion by SIGMA (conservative).

        This implements the geometric interpretation of chance
        constraints: direction uncertainty widens cones, so the
        safe region shrinks by the uncertainty budget.

        Parameters
        ----------
        SIGMA : float
            Angular uncertainty [rad] (1-sigma or confidence bound).
        """
        if self.sense == "inside":
            return Cap(self.c, max(0.0, self.ALPHA - SIGMA),
                       self.sense, self.label)
        else:
            return Cap(self.c, self.ALPHA + SIGMA,
                       self.sense, self.label)
