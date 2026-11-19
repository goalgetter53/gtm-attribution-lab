"""Rule-based (position / time) attribution models.

Each model maps one converted journey to a weight vector over its touches that sums to 1;
credits are the sum of those weights per channel across all converted journeys.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from gal.attribution.base import Credits, Journey, attributable

WeightFn = Callable[[Journey], npt.NDArray[np.float64]]


def first_touch_weights(j: Journey) -> npt.NDArray[np.float64]:
    w = np.zeros(len(j.channels))
    w[0] = 1.0
    return w


def last_touch_weights(j: Journey) -> npt.NDArray[np.float64]:
    w = np.zeros(len(j.channels))
    w[-1] = 1.0
    return w


def linear_weights(j: Journey) -> npt.NDArray[np.float64]:
    n = len(j.channels)
    return np.full(n, 1.0 / n)


def time_decay_weights(j: Journey, half_life_days: float = 7.0) -> npt.NDArray[np.float64]:
    """Exponential decay: a touch ``half_life_days`` older gets half the weight."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    d = np.asarray(j.days_before, dtype=np.float64)
    # Shift by the most recent touch for numerical stability (weights are re-normalised).
    w = np.power(0.5, (d - np.min(d)) / half_life_days)
    return np.asarray(w / w.sum(), dtype=np.float64)


def u_shaped_weights(j: Journey, edge: float = 0.4) -> npt.NDArray[np.float64]:
    """40% first touch, 40% last touch (lead-creation + conversion), 20% spread in between."""
    n = len(j.channels)
    if n == 1:
        return np.ones(1)
    if n == 2:
        return np.full(2, 0.5)
    w = np.full(n, (1.0 - 2 * edge) / (n - 2))
    w[0] = edge
    w[-1] = edge
    return w


def _milestone_index(j: Journey, milestone_days_before: float | None) -> int | None:
    """Last touch at or before the milestone (i.e. with days_before >= milestone's)."""
    if milestone_days_before is None:
        return None
    idx = [i for i, d in enumerate(j.days_before) if d >= milestone_days_before - 1e-9]
    return idx[-1] if idx else None


def w_shaped_weights(j: Journey, milestone: float = 0.3) -> npt.NDArray[np.float64]:
    """30% each to first touch, the MQL-creating touch and the opportunity-creating touch;
    the remaining 10% is spread over all other touches. Coinciding milestones add up;
    missing milestones or 'other' touches are renormalised away."""
    n = len(j.channels)
    w = np.zeros(n)
    milestones = [0, _milestone_index(j, j.mql_days_before), _milestone_index(j, j.opp_days_before)]
    for m in milestones:
        if m is not None:
            w[m] += milestone
    others = [i for i in range(n) if i not in {m for m in milestones if m is not None}]
    if others:
        w[others] += (1.0 - 3 * milestone) / len(others)
    return np.asarray(w / w.sum(), dtype=np.float64)


def attribute_with(journeys: Iterable[Journey], weight_fn: WeightFn) -> Credits:
    credits: Credits = {}
    for j in attributable(journeys):
        w = weight_fn(j)
        for ch, wi in zip(j.channels, w, strict=True):
            credits[ch] = credits.get(ch, 0.0) + float(wi)
    return credits


def first_touch(journeys: Iterable[Journey]) -> Credits:
    return attribute_with(journeys, first_touch_weights)


def last_touch(journeys: Iterable[Journey]) -> Credits:
    return attribute_with(journeys, last_touch_weights)


def linear(journeys: Iterable[Journey]) -> Credits:
    return attribute_with(journeys, linear_weights)


def time_decay(journeys: Iterable[Journey], half_life_days: float = 7.0) -> Credits:
    return attribute_with(journeys, lambda j: time_decay_weights(j, half_life_days))


def u_shaped(journeys: Iterable[Journey]) -> Credits:
    return attribute_with(journeys, u_shaped_weights)


def w_shaped(journeys: Iterable[Journey]) -> Credits:
    return attribute_with(journeys, w_shaped_weights)
