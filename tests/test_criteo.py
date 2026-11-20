"""Offline tests for the Criteo journey builder (tiny inline fixture, no network)."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from gal.attribution import heuristics, markov
from gal.real import criteo
from gal.real.compare import gap, percentile_ci, spearman_matrix

DAY = criteo.SECONDS_PER_DAY

# (timestamp, uid, campaign, conversion, conversion_timestamp, conversion_id, attribution, click)
# Campaign 7 has 6 impressions, 8 has 3, 9 has 2 -> with top_k=2: 7=c01, 8=c02, 9=other.
ROWS = [
    # user 1: view 7, click 8 -> conversion 100 at t=20 (Criteo-attributed); later view 7
    (0, 1, 7, 0, -1, -1, 0, 0),
    (10, 1, 8, 1, 20, 100, 1, 1),
    (30, 1, 7, 0, -1, -1, 0, 0),
    # user 2: two impressions of campaign 9 carry conversion 200 (not attributed to Criteo)
    (5, 2, 9, 1, 50, 200, 0, 0),
    (40, 2, 9, 1, 50, 200, 0, 1),
    # user 3: two views of 7, no conversion
    (1, 3, 7, 0, -1, -1, 0, 0),
    (2, 3, 7, 0, -1, -1, 0, 0),
    # user 4: one view of 8, no conversion
    (3, 4, 8, 0, -1, -1, 0, 0),
    # user 5: two conversions in a row, then a trailing view
    (1, 5, 7, 1, 5, 300, 1, 1),
    (6, 5, 8, 1, 8, 301, 0, 1),
    (9, 5, 7, 0, -1, -1, 0, 0),
]
COLS = [
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
]


def _journeys(rows: list[tuple[int, ...]], **kw: object) -> criteo.CriteoJourneys:
    con = duckdb.connect()
    criteo.register_frame(con, pd.DataFrame(rows, columns=COLS))
    return criteo.load_journeys(con, **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def cj() -> criteo.CriteoJourneys:
    return _journeys(ROWS, top_k=2)


def test_channels_are_top_k_plus_other(cj: criteo.CriteoJourneys) -> None:
    assert cj.channels == ("c01", "c02", "other")
    assert cj.stats["top_k_impression_share"] == pytest.approx(9 / 11)
    assert cj.stats["campaigns"] == 3


def test_converted_journeys_are_cut_at_each_conversion(cj: criteo.CriteoJourneys) -> None:
    paths = [j.channels for j in cj.converted]
    # user 1 (impressions before the conversion, from any campaign), user 2, user 5 x2
    assert paths == [("c01", "c02"), ("other", "other"), ("c01",), ("c02",)]
    assert all(j.converted for j in cj.converted)
    assert cj.converted[0].days_before == pytest.approx((20 / DAY, 10 / DAY))
    assert cj.converted[1].days_before == pytest.approx((45 / DAY, 10 / DAY))
    assert cj.converted_user.tolist() == [0, 1, 4, 4]
    assert cj.stats["conversions"] == 4
    assert cj.stats["converted_journeys_multi_channel"] == 1


def test_labels_owner_and_criteo(cj: criteo.CriteoJourneys) -> None:
    assert cj.owner_channel == ["c02", "other", "c01", "c02"]
    assert cj.criteo_channel == ["c02", None, "c01", None]
    assert criteo.label_counts(cj.criteo_channel, cj.channels) == {
        "c01": 1.0,
        "c02": 1.0,
        "other": 0.0,
    }
    assert cj.stats["conversions_criteo_attributed"] == 2


def test_nonconverting_paths_are_deduplicated(cj: criteo.CriteoJourneys) -> None:
    paths = {j.channels: n for j, n in zip(cj.nonconv_paths, cj.path_counts(), strict=True)}
    # trailing views of users 1 and 5 share one path; users 3 and 4 have their own
    assert paths == {("c01",): 2, ("c01", "c01"): 1, ("c02",): 1}
    assert not any(j.converted for j in cj.nonconv_paths)
    assert sorted(cj.nonconv_user.tolist()) == [0, 2, 3, 4]
    assert cj.stats["nonconverting_journeys"] == 4


def test_expand_feeds_the_repo_models(cj: criteo.CriteoJourneys) -> None:
    js = cj.expand()
    assert len(js) == 8
    assert sum(j.converted for j in js) == 4
    assert heuristics.last_touch(js) == {"c02": 2.0, "other": 1.0, "c01": 1.0}
    credits = markov.markov_attribution(js, order=1)
    assert sum(credits.values()) == pytest.approx(4.0)


def test_bootstrap_weights_whole_users(cj: criteo.CriteoJourneys) -> None:
    rng = np.random.default_rng(3)
    conv, paths = cj.bootstrap_counts(rng)
    w = np.random.default_rng(3).poisson(1.0, cj.n_users)
    assert conv.tolist() == w[cj.converted_user].tolist()
    assert int(paths.sum()) == int(w[cj.nonconv_user].sum())
    js = cj.expand(conv, paths)
    assert len(js) == int(conv.sum() + paths.sum())
    weighted = criteo.weighted_label_counts(cj.criteo_channel, conv, cj.channels)
    assert weighted["c02"] == conv[0] and weighted["c01"] == conv[2]


def test_touch_type_mode_labels_clicks_and_views() -> None:
    tj = _journeys(ROWS, top_k=2, channel_mode="touch_type")
    assert tj.channels == ("click", "view")
    assert tj.converted[0].channels == ("view", "click")
    assert tj.converted[1].channels == ("view", "click")


def test_same_timestamp_conversions_leave_one_empty_journey() -> None:
    rows = [(90, 6, 9, 1, 100, 400, 0, 1), (95, 6, 9, 1, 100, 401, 0, 1)]
    tj = _journeys(rows, top_k=1)
    assert tj.stats["conversions"] == 2
    assert tj.stats["converted_journeys_empty"] == 1
    assert sum(len(j.channels) for j in tj.converted) == 2


def test_invalid_arguments() -> None:
    con = duckdb.connect()
    criteo.register_frame(con, pd.DataFrame(ROWS, columns=COLS))
    with pytest.raises(ValueError):
        criteo.build_tables(con, top_k=0)
    with pytest.raises(ValueError):
        criteo.build_tables(con, channel_mode="bogus")  # type: ignore[arg-type]


def test_gap_and_rank_metrics() -> None:
    ch = ["a", "b", "c"]
    x = {"a": 0.5, "b": 0.3, "c": 0.2}
    y = {"a": 0.4, "b": 0.4, "c": 0.2}
    g = gap("x", x, "y", y, ch)
    assert g.max_abs_gap == pytest.approx(0.1)
    assert g.most_over == "a" and g.most_under == "b"
    assert g.total_variation == pytest.approx(0.1)
    m = spearman_matrix({"x": x, "y": y}, ch)
    assert m["x"]["x"] == pytest.approx(1.0)
    lo, hi = percentile_ci([float(v) for v in range(101)])
    assert (lo, hi) == pytest.approx((2.5, 97.5))
