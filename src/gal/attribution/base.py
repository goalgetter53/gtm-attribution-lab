"""Shared types for attribution models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

Credits = dict[str, float]
"""Channel -> credited conversions (fractional)."""


@dataclass(frozen=True)
class Journey:
    """One account's (or contact's) ordered touch path ending at a conversion or not.

    ``days_before[i]`` is how many days touch ``i`` happened before the conversion event
    (or before the end of observation for non-converting journeys); it is non-increasing.
    ``mql_days_before`` / ``opp_days_before`` locate the funnel milestones on the same
    clock (None if the milestone was never reached) and drive W-shaped attribution.
    """

    journey_id: str
    channels: tuple[str, ...]
    days_before: tuple[float, ...]
    converted: bool
    mql_days_before: float | None = None
    opp_days_before: float | None = None
    lead_days_before: float | None = None

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.days_before):
            raise ValueError("channels and days_before must have equal length")

    @property
    def is_empty(self) -> bool:
        return not self.channels


def attributable(journeys: Iterable[Journey]) -> list[Journey]:
    """Converted journeys with at least one touch: the conversions every model splits."""
    return [j for j in journeys if j.converted and not j.is_empty]


def total_attributable(journeys: Iterable[Journey]) -> int:
    return len(attributable(journeys))


def channels_of(journeys: Iterable[Journey]) -> list[str]:
    return sorted({c for j in journeys for c in j.channels})


def to_shares(credits: Credits, channels: Sequence[str]) -> dict[str, float]:
    """Normalise credits over ``channels`` only (e.g. drop the untracked 'direct' bucket)."""
    total = sum(max(credits.get(c, 0.0), 0.0) for c in channels)
    if total <= 0:
        return dict.fromkeys(channels, 0.0)
    return {c: max(credits.get(c, 0.0), 0.0) / total for c in channels}


def match_horizons(journeys: Sequence[Journey], rng: np.random.Generator) -> list[Journey]:
    """Risk-set matching of observation horizons for non-converting journeys.

    Converted paths stop at the win, while non-converting paths keep collecting touches
    until the data cutoff. Comparing them directly penalises channels that tend to happen
    late. Each non-converting journey is truncated at a horizon (days since lead creation)
    drawn from the empirical lead->win durations of converted journeys. Journeys without
    ``lead_days_before`` and all converted journeys are returned unchanged.
    """
    durations = [
        j.lead_days_before for j in journeys if j.converted and j.lead_days_before is not None
    ]
    if not durations:
        return list(journeys)
    pool = np.asarray(durations, dtype=np.float64)
    out: list[Journey] = []
    for j in journeys:
        if j.converted or j.lead_days_before is None:
            out.append(j)
            continue
        horizon = float(rng.choice(pool))
        cut = j.lead_days_before - horizon  # days_before value of the horizon
        keep = [i for i, d in enumerate(j.days_before) if d >= cut]
        out.append(
            replace(
                j,
                channels=tuple(j.channels[i] for i in keep),
                days_before=tuple(j.days_before[i] - cut for i in keep),
                lead_days_before=horizon,
            )
        )
    return out
