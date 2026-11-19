"""Funnel analytics: cohort conversion, pipeline velocity and Kaplan-Meier time-to-stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

STAGE_COLUMNS: dict[str, str] = {
    "mql": "mql_at",
    "sql": "sql_at",
    "opportunity": "opportunity_at",
    "won": "won_at",
}


def pipeline_velocity(
    n_opportunities: float, win_rate: float, avg_acv: float, avg_cycle_days: float
) -> float:
    """Pipeline velocity = (# opportunities x win rate x average ACV) / sales-cycle days.

    Interpreted as expected closed-won revenue per day flowing through the pipeline.
    """
    if avg_cycle_days <= 0:
        raise ValueError("avg_cycle_days must be positive")
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    return n_opportunities * win_rate * avg_acv / avg_cycle_days


@dataclass(frozen=True)
class KaplanMeier:
    """Product-limit survival estimate with Greenwood 95% (log-log) confidence bands."""

    times: FloatArray  # distinct event times (0 prepended)
    survival: FloatArray
    ci_low: FloatArray
    ci_high: FloatArray
    at_risk: npt.NDArray[np.int64]
    events: npt.NDArray[np.int64]

    def survival_at(self, t: float) -> float:
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        return float(self.survival[max(idx, 0)])

    def median(self) -> float | None:
        below = np.nonzero(self.survival <= 0.5)[0]
        return float(self.times[below[0]]) if below.size else None


def kaplan_meier(durations: Sequence[float], observed: Sequence[bool]) -> KaplanMeier:
    """Kaplan-Meier estimator.

    Args:
        durations: time from origin to event or censoring (>= 0).
        observed: True if the event happened at ``duration``; False if right-censored.
    """
    d = np.asarray(durations, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)
    if d.shape != e.shape:
        raise ValueError("durations and observed must have the same shape")
    if d.size and float(np.min(d)) < 0:
        raise ValueError("durations must be non-negative")
    event_times = np.unique(d[e])
    times = [0.0]
    surv = [1.0]
    lo = [1.0]
    hi = [1.0]
    at_risk_l = [int(d.size)]
    events_l = [0]
    s = 1.0
    greenwood = 0.0
    for t in event_times:
        n_risk = int((d >= t).sum())
        n_evt = int(((d == t) & e).sum())
        s *= 1.0 - n_evt / n_risk
        if n_risk > n_evt:
            greenwood += n_evt / (n_risk * (n_risk - n_evt))
        times.append(float(t))
        surv.append(s)
        at_risk_l.append(n_risk)
        events_l.append(n_evt)
        if 0.0 < s < 1.0 and greenwood > 0:
            # log(-log S) transform keeps the interval inside (0, 1).
            se = np.sqrt(greenwood) / abs(np.log(s))
            lo.append(float(s ** np.exp(1.96 * se)))
            hi.append(float(s ** np.exp(-1.96 * se)))
        else:
            lo.append(s)
            hi.append(s)
    return KaplanMeier(
        times=np.asarray(times),
        survival=np.asarray(surv),
        ci_low=np.asarray(lo),
        ci_high=np.asarray(hi),
        at_risk=np.asarray(at_risk_l, dtype=np.int64),
        events=np.asarray(events_l, dtype=np.int64),
    )


def time_to_stage(funnel: pd.DataFrame, stage: str, data_end: pd.Timestamp) -> KaplanMeier:
    """Days from lead creation to ``stage``; accounts not there yet are censored at cutoff.

    Note: accounts that are effectively dead (e.g. closed-lost) are also treated as censored,
    so ``1 - S(t)`` is an upper bound on the true reach rate under competing risks.
    """
    col = STAGE_COLUMNS[stage]
    lead = pd.to_datetime(funnel["lead_at"])
    reached = pd.to_datetime(funnel[col])
    observed = reached.notna()
    end = reached.where(observed, data_end)
    days = (end - lead).dt.total_seconds() / 86400.0
    return kaplan_meier(days.clip(lower=0).to_numpy(), observed.to_numpy())


def funnel_summary(con: Any) -> dict[str, pd.DataFrame]:
    """Pull the funnel marts into DataFrames."""
    return {
        "cohorts": con.execute("select * from marts.mart_funnel_cohorts").df(),
        "velocity": con.execute("select * from marts.mart_pipeline_velocity").df(),
        "funnel": con.execute("select * from marts.fct_funnel_stages").df(),
        "identity": con.execute("select * from marts.mart_identity_coverage").df(),
        "channels": con.execute("select * from marts.mart_channel_summary").df(),
    }
