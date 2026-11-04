"""Binomial confidence intervals for small-sample rates.

Agent evaluations usually have tens of runs per condition, where the normal
(Wald) interval is badly behaved: it can extend past 0 or 1 and collapses to
zero width when every run passes. The Wilson score interval stays inside
[0, 1] and keeps sensible coverage at small n, so it is used throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    estimate: float
    low: float
    high: float
    confidence: float

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a (n=0)"
        return f"{self.estimate:.1%} [{self.low:.1%}, {self.high:.1%}] ({self.successes}/{self.n})"


def wilson(successes: int, n: int, confidence: float = 0.95) -> Proportion:
    """Wilson score interval for ``successes`` out of ``n`` trials."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return Proportion(0, 0, math.nan, math.nan, math.nan, confidence)

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # Clamp floating-point drift at the boundaries (e.g. -1e-17).
    return Proportion(successes, n, p, max(0.0, centre - half), min(1.0, centre + half), confidence)
