from __future__ import annotations

import pytest

from gal.attribution.base import Journey
from gal.attribution.markov import fit_chain, markov_attribution, removal_effects, state_sequence


def j(chs: str, conv: bool) -> Journey:
    return Journey(chs, tuple(chs), tuple(float(len(chs) - i) for i in range(len(chs))), conv)


def test_hand_computed_removal_effects() -> None:
    # Paths: A->B->conv, A->null, B->conv. P(conv) from start:
    # start->A 2/3, start->B 1/3; A->B 1/2, A->null 1/2; B->conv 1.
    # P = 2/3*1/2 + 1/3 = 2/3. Remove A: 1/3 -> RE 0.5. Remove B: 0 -> RE 1.
    js = [j("AB", True), j("A", False), j("B", True)]
    chain = fit_chain(js)
    assert chain.conversion_probability() == pytest.approx(2 / 3)
    re = removal_effects(js)
    assert re == pytest.approx({"A": 0.5, "B": 1.0})
    credits = markov_attribution(js)
    assert credits["A"] == pytest.approx(2 * 0.5 / 1.5)
    assert sum(credits.values()) == pytest.approx(2.0)


def test_state_sequence_higher_order() -> None:
    assert state_sequence(("a", "b", "c"), 2) == [("a",), ("a", "b"), ("b", "c")]


def test_second_order_runs_and_sums() -> None:
    js = [j("ABC", True), j("ACB", False), j("BBA", True), j("C", False)]
    credits = markov_attribution(js, order=2)
    assert sum(credits.values()) == pytest.approx(2.0)


def test_self_loops_are_solvable() -> None:
    js = [j("AAAA", True), j("AA", False)]
    assert removal_effects(js) == {"A": pytest.approx(1.0)}


def test_no_conversions_gives_zero() -> None:
    js = [j("AB", False)]
    assert removal_effects(js) == {"A": 0.0, "B": 0.0}
    assert markov_attribution(js) == {"A": 0.0, "B": 0.0}


def test_invalid_order() -> None:
    with pytest.raises(ValueError):
        fit_chain([j("A", True)], order=0)
