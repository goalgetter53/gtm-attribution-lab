from __future__ import annotations

import numpy as np
import pytest

from gal.attribution import heuristics as h
from gal.attribution.base import Journey, attributable, match_horizons, to_shares

J = Journey("a", ("email", "paid", "webinar", "sdr"), (30.0, 20.0, 10.0, 0.0), True)


def test_first_and_last_touch() -> None:
    assert h.first_touch([J]) == {"email": 1.0, "paid": 0.0, "webinar": 0.0, "sdr": 0.0}
    assert h.last_touch([J])["sdr"] == 1.0


def test_linear_splits_evenly() -> None:
    assert h.linear([J]) == dict.fromkeys(J.channels, 0.25)


def test_time_decay_half_life() -> None:
    w = h.time_decay_weights(J, half_life_days=10.0)
    # each step back is one half-life: weights 1/8, 1/4, 1/2, 1 (normalised)
    assert np.allclose(w, np.array([1, 2, 4, 8]) / 15)
    with pytest.raises(ValueError):
        h.time_decay_weights(J, 0)


def test_u_shaped_edges() -> None:
    assert np.allclose(h.u_shaped_weights(J), [0.4, 0.1, 0.1, 0.4])
    two = Journey("b", ("x", "y"), (2.0, 1.0), True)
    assert np.allclose(h.u_shaped_weights(two), [0.5, 0.5])


def test_w_shaped_milestones() -> None:
    j = Journey("c", J.channels, J.days_before, True, mql_days_before=18.0, opp_days_before=5.0)
    # first touch (idx0), MQL touch = last touch with days_before >= 18 (idx1),
    # opp touch = last with days_before >= 5 (idx2); 10% to the remaining touch (idx3)
    assert np.allclose(h.w_shaped_weights(j), [0.3, 0.3, 0.3, 0.1])


def test_w_shaped_coinciding_milestones_renormalise() -> None:
    j = Journey("d", ("x", "y"), (10.0, 1.0), True, mql_days_before=50.0, opp_days_before=50.0)
    w = h.w_shaped_weights(j)
    assert w.sum() == pytest.approx(1.0) and w[0] > w[1]


def test_non_converted_and_empty_journeys_ignored() -> None:
    js = [J, Journey("e", ("x",), (1.0,), False), Journey("f", (), (), True)]
    assert len(attributable(js)) == 1
    assert sum(h.linear(js).values()) == pytest.approx(1.0)


def test_journey_validates_lengths() -> None:
    with pytest.raises(ValueError):
        Journey("g", ("x",), (), True)


def test_to_shares_excludes_untracked() -> None:
    s = to_shares({"a": 2.0, "b": 2.0, "direct": 4.0}, ["a", "b"])
    assert s == {"a": 0.5, "b": 0.5}
    assert to_shares({}, ["a"]) == {"a": 0.0}


def test_match_horizons_truncates_only_non_converters() -> None:
    conv = Journey("c", ("x",), (0.0,), True, lead_days_before=10.0)
    lost = Journey("n", ("x", "y", "z"), (90.0, 50.0, 5.0), False, lead_days_before=100.0)
    out = match_horizons([conv, lost], np.random.default_rng(0))
    assert out[0] == conv
    # horizon must be 10 days after lead: only touches at days-since-lead <= 10 survive
    assert out[1].channels == ("x",)
    assert out[1].days_before == pytest.approx((0.0,))
