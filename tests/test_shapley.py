from __future__ import annotations

import numpy as np
import pytest

from gal.attribution.base import Journey
from gal.attribution.shapley import (
    empirical_rates,
    fit_logistic,
    mc_convergence,
    shapley_empirical,
    shapley_exact,
    shapley_from_table,
    shapley_model,
    shapley_monte_carlo,
)


def glove(s: frozenset[str]) -> float:
    # one left glove (L), two right gloves (R1, R2); a pair is worth 1
    return float("L" in s and ("R1" in s or "R2" in s))


def test_glove_game_textbook_values() -> None:
    phi = shapley_exact(["L", "R1", "R2"], glove)
    assert phi == pytest.approx({"L": 2 / 3, "R1": 1 / 6, "R2": 1 / 6})


def test_additive_game_returns_weights() -> None:
    w = {"a": 1.0, "b": 2.5, "c": -0.5}
    phi = shapley_exact(list(w), lambda s: sum(w[p] for p in s))
    assert phi == pytest.approx(w)


def test_table_must_be_power_of_two() -> None:
    with pytest.raises(ValueError):
        shapley_from_table(np.zeros(6))


def test_monte_carlo_close_to_exact() -> None:
    rng = np.random.default_rng(0)
    mc = shapley_monte_carlo(["L", "R1", "R2"], glove, 4000, rng)
    assert mc.values == pytest.approx({"L": 2 / 3, "R1": 1 / 6, "R2": 1 / 6}, abs=0.03)
    assert all(se < 0.02 for se in mc.std_errors.values())


def test_monte_carlo_convergence_error_shrinks() -> None:
    players = [f"p{i}" for i in range(7)]
    rng0 = np.random.default_rng(42)
    weights = rng0.random(7)

    def v(s: frozenset[str]) -> float:
        x = sum(weights[int(p[1:])] for p in s)
        return float(x**2)  # non-additive (interactions)

    pts = mc_convergence(players, v, [20, 2000], np.random.default_rng(1), repeats=4)
    assert pts[1][1] < pts[0][1] / 4  # 100x samples -> ~10x less error


def test_empirical_rates_smoothing() -> None:
    js = [Journey("1", ("a",), (1.0,), True), Journey("2", ("a",), (1.0,), False)]
    v = empirical_rates(js, prior_strength=0.0)
    assert v(frozenset()) == 0.0
    assert v(frozenset({"a"})) == pytest.approx(0.5)


def test_logistic_irls_recovers_coefficients() -> None:
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(20000), rng.normal(size=(20000, 2))])
    beta = np.array([-1.0, 0.8, -0.5])
    y = (rng.random(20000) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    est = fit_logistic(X, y, l2=0.0)
    assert est == pytest.approx(beta, abs=0.06)


def test_shapley_model_ranks_effective_channel_first() -> None:
    rng = np.random.default_rng(5)
    js = []
    for i in range(3000):
        n_good, n_noise = rng.poisson(1.0), rng.poisson(1.0)
        chs = ("good",) * n_good + ("noise",) * n_noise
        p = 1 - 0.7**n_good * 0.98**n_noise
        js.append(Journey(str(i), chs, tuple(range(len(chs), 0, -1)), bool(rng.random() < p)))
    credits = shapley_model(js)
    assert credits["good"] > 5 * credits["noise"]
    assert sum(credits.values()) == pytest.approx(sum(j.converted and bool(j.channels) for j in js))


def test_shapley_empirical_uses_mc_when_many_channels() -> None:
    js = [Journey(str(i), tuple("abcdefg"), tuple(range(7, 0, -1)), i % 2 == 0) for i in range(10)]
    exact = shapley_empirical(js, exact_max_channels=10)
    mc = shapley_empirical(js, exact_max_channels=3, n_permutations=10000)
    assert sum(mc.values()) == pytest.approx(5.0)
    assert mc == pytest.approx(exact, abs=0.07)


def test_exact_player_limit() -> None:
    with pytest.raises(ValueError):
        shapley_exact([str(i) for i in range(21)], lambda s: 0.0)
