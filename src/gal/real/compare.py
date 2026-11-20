"""Model-vs-model disagreement metrics for data without ground truth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from gal.attribution.base import Credits, to_shares
from gal.stats import spearman

Shares = dict[str, float]


def shares_by_model(credits: Mapping[str, Credits], channels: Sequence[str]) -> dict[str, Shares]:
    return {m: to_shares(c, channels) for m, c in credits.items()}


@dataclass(frozen=True)
class Gap:
    """How far model ``a`` is from model ``b`` (shares, a minus b)."""

    a: str
    b: str
    max_abs_gap: float
    max_abs_gap_channel: str
    most_over: str  # channel where a - b is largest
    most_over_gap: float
    most_under: str  # channel where a - b is smallest
    most_under_gap: float
    spearman: float
    total_variation: float  # 0.5 * sum |a - b|: share of credit that would have to move


def gap(a_name: str, a: Shares, b_name: str, b: Shares, channels: Sequence[str]) -> Gap:
    d = {c: a.get(c, 0.0) - b.get(c, 0.0) for c in channels}
    worst = max(channels, key=lambda c: abs(d[c]))
    over = max(channels, key=lambda c: d[c])
    under = min(channels, key=lambda c: d[c])
    return Gap(
        a=a_name,
        b=b_name,
        max_abs_gap=abs(d[worst]),
        max_abs_gap_channel=worst,
        most_over=over,
        most_over_gap=d[over],
        most_under=under,
        most_under_gap=d[under],
        spearman=spearman([a.get(c, 0.0) for c in channels], [b.get(c, 0.0) for c in channels]),
        total_variation=0.5 * float(np.sum(np.abs(list(d.values())))),
    )


def spearman_matrix(
    shares: Mapping[str, Shares], channels: Sequence[str]
) -> dict[str, dict[str, float]]:
    names = list(shares)
    vec = {m: [shares[m].get(c, 0.0) for c in channels] for m in names}
    return {a: {b: spearman(vec[a], vec[b]) for b in names} for a in names}


def percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.shape[0] == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
