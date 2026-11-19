"""Hypothesis property tests for the attribution axioms."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gal.attribution import heuristics as h
from gal.attribution.base import Journey, total_attributable
from gal.attribution.markov import markov_attribution, removal_effects
from gal.attribution.shapley import shapley_empirical, shapley_exact, shapley_model

CHANNELS = ["email", "paid", "webinar", "sdr", "events"]


@st.composite
def journeys(draw: st.DrawFn) -> list[Journey]:
    n = draw(st.integers(min_value=1, max_value=25))
    out = []
    for i in range(n):
        k = draw(st.integers(min_value=0, max_value=6))
        chs = tuple(draw(st.sampled_from(CHANNELS)) for _ in range(k))
        gaps = draw(st.lists(st.floats(0, 30, allow_nan=False), min_size=k, max_size=k))
        days = tuple(float(x) for x in np.cumsum(gaps[::-1])[::-1]) if k else ()
        mql = draw(st.none() | st.floats(0, 120, allow_nan=False))
        opp = draw(st.none() | st.floats(0, 120, allow_nan=False))
        out.append(Journey(str(i), chs, days, draw(st.booleans()), mql, opp))
    return out


HEURISTICS = [
    h.first_touch,
    h.last_touch,
    h.linear,
    h.u_shaped,
    h.w_shaped,
    lambda js: h.time_decay(js, 7.0),
]


@settings(max_examples=60, deadline=None)
@given(journeys())
def test_heuristic_credits_sum_to_conversions(js: list[Journey]) -> None:
    n = total_attributable(js)
    for model in HEURISTICS:
        credits = model(js)
        assert sum(credits.values()) == pytest.approx(n, abs=1e-9)
        assert all(v >= -1e-12 for v in credits.values())


@settings(max_examples=60, deadline=None)
@given(journeys(), st.integers(min_value=1, max_value=3))
def test_markov_removal_effects_non_negative_and_credits_sum(js: list[Journey], order: int) -> None:
    re = removal_effects(js, order)
    assert all(0.0 <= v <= 1.0 for v in re.values())
    credits = markov_attribution(js, order)
    n = total_attributable(js)
    if n:
        assert sum(credits.values()) == pytest.approx(n)


@settings(max_examples=40, deadline=None)
@given(journeys())
def test_shapley_attribution_credits_sum_to_conversions(js: list[Journey]) -> None:
    n = total_attributable(js)
    assert sum(shapley_empirical(js).values()) == pytest.approx(n)
    assert sum(shapley_model(js).values()) == pytest.approx(n)


game_values = st.lists(st.floats(-10, 10, allow_nan=False), min_size=16, max_size=16)


def _game(vals: list[float], players: list[str]) -> dict[frozenset[str], float]:
    table = {}
    for mask in range(1 << len(players)):
        table[frozenset(p for i, p in enumerate(players) if mask >> i & 1)] = vals[mask]
    return table


@settings(max_examples=80, deadline=None)
@given(game_values)
def test_shapley_efficiency_axiom(vals: list[float]) -> None:
    players = ["a", "b", "c", "d"]
    g = _game(vals, players)
    phi = shapley_exact(players, g.__getitem__)
    assert sum(phi.values()) == pytest.approx(g[frozenset(players)] - g[frozenset()], abs=1e-9)


@settings(max_examples=80, deadline=None)
@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=5, max_size=5))
def test_shapley_symmetry_and_null_player(by_size: list[float]) -> None:
    # v depends only on how many of {a, b, c} are present -> a, b, c are symmetric;
    # d never changes the value -> d is a null player.
    players = ["a", "b", "c", "d"]

    def v(s: frozenset[str]) -> float:
        return by_size[len(s - {"d"})]

    phi = shapley_exact(players, v)
    assert phi["a"] == pytest.approx(phi["b"]) and phi["b"] == pytest.approx(phi["c"])
    assert phi["d"] == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(game_values, game_values)
def test_shapley_linearity(v1: list[float], v2: list[float]) -> None:
    players = ["a", "b", "c", "d"]
    g1, g2 = _game(v1, players), _game(v2, players)
    p1 = shapley_exact(players, g1.__getitem__)
    p2 = shapley_exact(players, g2.__getitem__)
    p12 = shapley_exact(players, lambda s: g1[s] + g2[s])
    for p in players:
        assert p12[p] == pytest.approx(p1[p] + p2[p], abs=1e-9)


def test_shapley_matches_permutation_definition() -> None:
    rng = np.random.default_rng(9)
    players = ["a", "b", "c", "d"]
    g = _game(list(rng.normal(size=16)), players)
    phi = shapley_exact(players, g.__getitem__)
    perms = list(itertools.permutations(players))
    for p in players:
        total = 0.0
        for order in perms:
            before = frozenset(order[: order.index(p)])
            total += g[before | {p}] - g[before]
        assert phi[p] == pytest.approx(total / len(perms))
