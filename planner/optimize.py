"""SNR optimization: find best (B, N) pair given umbra budget."""

from dataclasses import dataclass

from constants import DEFAULT_STARTUP_S, DEFAULT_N_PIXELS, DEFAULT_READ_RATE


@dataclass
class Instrument:
    """CCD instrument parameters."""
    startup_s: float = DEFAULT_STARTUP_S
    n_pixels: int = DEFAULT_N_PIXELS
    read_rate: int = DEFAULT_READ_RATE

    @property
    def readout_per_sample(self) -> float:
        """Time (s) for one full-frame readout."""
        return self.n_pixels / self.read_rate


@dataclass
class Solution:
    """Optimal imaging parameters for a single eclipse pass."""
    n_samples: int
    exposure_s: float
    objective: float
    umbra_s: float
    usable_s: float


def optimize(
    umbra_s: float,
    inst: Instrument,
    n_max: int = 20,
) -> Solution | None:
    """Find (B, N) maximizing B * N within the umbra budget.

    Returns None if no feasible solution exists (umbra too short).
    """
    usable = umbra_s - inst.startup_s
    if usable <= 0:
        return None

    dt_read = inst.readout_per_sample
    best = None

    for n in range(1, n_max + 1):
        b = usable - dt_read * n
        if b <= 0:
            break
        obj = b * n
        if best is None or obj > best.objective:
            best = Solution(
                n_samples=n,
                exposure_s=b,
                objective=obj,
                umbra_s=umbra_s,
                usable_s=usable,
            )

    return best
