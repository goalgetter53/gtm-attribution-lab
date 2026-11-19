from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gal.funnel import kaplan_meier, pipeline_velocity, time_to_stage
from gal.pipeline import PipelineRun
from gal.stats import (
    bootstrap_interval,
    mean_abs_error,
    rankdata,
    spearman,
    t_critical_975,
    t_interval,
)


def test_kaplan_meier_textbook_example() -> None:
    # durations 1,2,2,3,4 ; censored at 3 -> S: t=1 4/5, t=2 4/5*2/4, t=4 0
    km = kaplan_meier([1, 2, 2, 3, 4], [True, True, True, False, True])
    assert km.times.tolist() == [0.0, 1.0, 2.0, 4.0]
    assert km.survival.tolist() == pytest.approx([1.0, 0.8, 0.4, 0.0])
    assert km.at_risk.tolist() == [5, 5, 4, 1]
    assert km.median() == 2.0
    assert km.survival_at(3.5) == pytest.approx(0.4)


def test_kaplan_meier_all_censored_and_ci_bounds() -> None:
    km = kaplan_meier([5, 6, 7], [False, False, False])
    assert km.survival.tolist() == [1.0] and km.median() is None
    rng = np.random.default_rng(0)
    d = rng.exponential(10, 300)
    km2 = kaplan_meier(d, rng.random(300) < 0.7)
    assert np.all(km2.ci_low <= km2.survival + 1e-12)
    assert np.all(km2.survival <= km2.ci_high + 1e-12)
    assert np.all(np.diff(km2.survival) <= 0)


def test_kaplan_meier_input_validation() -> None:
    with pytest.raises(ValueError):
        kaplan_meier([1, -1], [True, True])
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [True])


def test_pipeline_velocity_formula() -> None:
    # 100 opps x 25% x $40k / 80 days = $12,500/day
    assert pipeline_velocity(100, 0.25, 40_000, 80) == pytest.approx(12_500)
    with pytest.raises(ValueError):
        pipeline_velocity(10, 0.5, 1, 0)
    with pytest.raises(ValueError):
        pipeline_velocity(10, 1.5, 1, 10)


def test_velocity_mart_matches_python_formula(small_run: PipelineRun) -> None:
    row = (
        small_run.con.execute("select * from marts.mart_pipeline_velocity where grain = 'all_time'")
        .df()
        .iloc[0]
    )
    expected = pipeline_velocity(row.n_opportunities, row.win_rate, row.avg_acv, row.avg_cycle_days)
    assert row.velocity_per_day == pytest.approx(expected)


def test_cohort_mart_consistent_with_funnel(small_run: PipelineRun) -> None:
    con = small_run.con
    cohorts = con.execute("select * from marts.mart_funnel_cohorts").df()
    funnel = con.execute("select * from marts.fct_funnel_stages").df()
    assert cohorts.leads.sum() == len(funnel)
    assert cohorts.wins.sum() == funnel.won_at.notna().sum()


def test_time_to_stage_censors_unreached(small_run: PipelineRun) -> None:
    funnel = small_run.con.execute("select * from marts.fct_funnel_stages").df()
    km = time_to_stage(funnel, "won", pd.Timestamp("2026-03-27"))
    assert km.events.sum() == funnel.won_at.notna().sum()
    assert 0 < 1 - km.survival[-1] < 1


def test_rankdata_and_spearman() -> None:
    assert rankdata([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert mean_abs_error([0.5, 0.5], [0.2, 0.8]) == pytest.approx(0.3)


def test_t_interval() -> None:
    iv = t_interval([1.0, 2.0, 3.0, float("nan")])
    assert iv.n == 3 and iv.mean == pytest.approx(2.0)
    assert iv.high - iv.mean == pytest.approx(t_critical_975(2) * 1.0 / np.sqrt(3))
    assert t_critical_975(200) == 1.96
    assert t_interval([4.0]).low == 4.0


def test_bootstrap_interval_covers_mean() -> None:
    rng = np.random.default_rng(0)
    iv = bootstrap_interval(list(rng.normal(5, 1, 200)), rng)
    assert iv.low < 5 < iv.high
