"""Boolean region on S² from intersection of Caps.

A(t) = intersection of constraint caps, evaluated on a Fibonacci mesh.
Provides: feasible mask, margin field, dominance map, uncertainty.
"""

from dataclasses import dataclass, field
import numpy as np
from . import vec
from .cap import Cap


@dataclass(slots=True)
class Region:
    """Intersection of spherical caps evaluated on a mesh.

    Parameters
    ----------
    caps : list[Cap]
        Constraint regions to intersect.
    mesh : ndarray, shape (n, 3)
        Sphere sample points.
    """
    caps: list = field(default_factory=list)
    mesh: np.ndarray = field(default_factory=lambda: vec.sphere(2000))

    _mask: np.ndarray = field(init=False, repr=False, default=None)
    _margins: np.ndarray = field(init=False, repr=False, default=None)
    _dominant: np.ndarray = field(init=False, repr=False, default=None)

    def solve(self):
        """Compute feasible mask, margins, and dominance over mesh."""
        n = len(self.mesh)
        m = len(self.caps)

        # per-cap margin matrix: (m, n)
        M = np.empty((m, n))
        for i, cap in enumerate(self.caps):
            M[i] = cap.margin(self.mesh)

        # feasible = all margins positive
        self._mask = np.all(M >= 0, axis=0)

        # min margin across caps at each point
        self._margins = np.min(M, axis=0)

        # dominant constraint = index of smallest margin (tightest)
        self._dominant = np.argmin(M, axis=0)

        return self

    # --- properties ---

    @property
    def mask(self):
        if self._mask is None:
            self.solve()
        return self._mask

    @property
    def margins(self):
        if self._margins is None:
            self.solve()
        return self._margins

    @property
    def dominant(self):
        """Per-point index of tightest constraint."""
        if self._dominant is None:
            self.solve()
        return self._dominant

    # --- metrics ---

    def exists(self):
        """True if any feasible attitude exists."""
        return bool(np.any(self.mask))

    def area(self):
        """Feasible solid angle [sr]."""
        return 4 * np.pi * np.mean(self.mask)

    def fraction(self):
        """Feasible fraction of sphere [0, 1]."""
        return float(np.mean(self.mask))

    def center(self):
        """Spherical mean of feasible points. None if empty."""
        if not self.exists():
            return None
        return vec.hat(self.mesh[self.mask].mean(axis=0))

    def maxmargin(self):
        """Safest attitude: max of min-margin.

        Returns (unit vector, margin) or (None, -inf).
        """
        idx = np.argmax(self.margins)
        if self.margins[idx] <= 0:
            return None, float('-inf')
        return vec.hat(self.mesh[idx]), float(self.margins[idx])

    def pointmargin(self, b):
        """Min margin at a specific direction b [rad]."""
        return float(min(cap.margin(b) for cap in self.caps))

    # --- dominance ---

    def dominance(self):
        """Constraint-dominance: label -> fraction where tightest."""
        labels = [cap.label or f"c{i}" for i, cap in enumerate(self.caps)]
        counts = np.bincount(self.dominant, minlength=len(self.caps))
        total = len(self.mesh)
        return {labels[i]: counts[i] / total for i in range(len(self.caps))}

    def bottleneck(self):
        """Label of the globally dominant constraint."""
        d = self.dominance()
        return max(d, key=d.get)

    # --- uncertainty ---

    def uncertain(self, SIGMA):
        """Return a new Region with all caps widened by SIGMA.

        Parameters
        ----------
        SIGMA : float or dict
            If float: uniform angular uncertainty [rad] for all caps.
            If dict: maps cap.label -> uncertainty [rad].
        """
        if isinstance(SIGMA, dict):
            widened = []
            for cap in self.caps:
                s = SIGMA.get(cap.label, 0.0)
                widened.append(cap.widen(s))
        else:
            widened = [cap.widen(SIGMA) for cap in self.caps]
        return Region(caps=widened, mesh=self.mesh).solve()
