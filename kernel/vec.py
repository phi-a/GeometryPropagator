"""Unit-vector arithmetic on R³ and S²."""

import numpy as np

# --- constructors ---

def hat(v):
    """Normalize to unit vector."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def radec(ALPHA, DELTA):
    """Unit vector from right ascension and declination (radians)."""
    return np.array([
        np.cos(ALPHA) * np.cos(DELTA),
        np.sin(ALPHA) * np.cos(DELTA),
        np.sin(DELTA),
    ])


def sphere(n=2000):
    """Fibonacci-lattice points on S², shape (n, 3)."""
    PHI = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    THETA = 2 * np.pi * i / PHI
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(THETA), r * np.sin(THETA), z])


# --- metrics ---

def dot(a, b):
    """Dot product, last axis."""
    return np.sum(a * b, axis=-1)


def angle(a, b):
    """Great-circle angle between unit vectors (radians)."""
    return np.arccos(np.clip(dot(a, b), -1.0, 1.0))
