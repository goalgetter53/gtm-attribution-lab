"""Small, dependency-free statistics helpers (ranks, Spearman, t-intervals, bootstrap)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Two-sided 95% Student-t critical values t_{0.975, df} for df = 1..30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("df must be >= 1")
    return _T975[df - 1] if df <= len(_T975) else 1.96


def rankdata(x: Sequence[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """1-based ranks with ties replaced by their average rank."""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0])
    ranks[order] = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    for v in np.unique(a):
        tied = a == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; NaN if either input is constant."""
    if len(x) != len(y):
        raise ValueError("inputs must have the same length")
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_abs_error(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def t_interval(values: Sequence[float]) -> Interval:
    """95% t confidence interval of the mean (ignores NaNs)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    n = int(v.shape[0])
    if n == 0:
        nan = float("nan")
        return Interval(nan, nan, nan, 0)
    m = float(v.mean())
    if n == 1:
        return Interval(m, m, m, 1)
    half = t_critical_975(n - 1) * float(v.std(ddof=1)) / np.sqrt(n)
    return Interval(m, m - half, m + half, n)


def bootstrap_interval(
    values: Sequence[float], rng: np.random.Generator, n_boot: int = 2000
) -> Interval:
    """Percentile bootstrap 95% interval of the mean."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.shape[0] == 0:
        nan = float("nan")
        return Interval(nan, nan, nan, 0)
    idx = rng.integers(0, v.shape[0], (n_boot, v.shape[0]))
    means = v[idx].mean(axis=1)
    return Interval(
        float(v.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        int(v.shape[0]),
    )
