"""Shapley-value attribution.

Core game theory
----------------
``shapley_exact`` enumerates all ``2^n`` coalitions (bitmask table, vectorised) and is used
for up to ``exact_max_channels`` players. ``shapley_monte_carlo`` samples random player
orderings (Castro et al., 2009) and returns standard errors; it converges to the exact
value at ``O(1/sqrt(permutations))``.

Two characteristic functions for attribution
--------------------------------------------
* ``empirical``: v(S) = smoothed conversion rate of journeys whose *set* of channels is
  exactly S (Beta prior centred on the rate of journeys with |S| channels). This is the
  classic data-driven Shapley used by many MTA tools.
* ``model``: v_j(S) = P(convert | only the channels in S kept for journey j), from a ridge
  logistic regression on touch counts per channel (counts, not log-counts, because a
  noisy-OR response is log-linear in counts). This is SHAP-style Shapley over a
  conversion-response model and uses touch frequency, not just presence.

For each converted journey the per-journey Shapley values are clipped at 0 and normalised,
so every attributable conversion is split exactly once (credits sum to conversions).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import factorial

import numpy as np
import numpy.typing as npt

from gal.attribution.base import Credits, Journey, attributable

FloatArray = npt.NDArray[np.float64]
Value = Callable[[frozenset[str]], float]


# ------------------------------------------------------------------- core Shapley


def _popcount(masks: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    counts = np.zeros_like(masks)
    m = masks.copy()
    while m.any():
        counts += m & 1
        m >>= 1
    return counts


def shapley_from_table(values: FloatArray) -> FloatArray:
    """Exact Shapley values from a table ``values[mask] = v(S)`` with ``len = 2**n``."""
    size = values.shape[0]
    n = size.bit_length() - 1
    if 1 << n != size:
        raise ValueError("value table length must be a power of two")
    masks = np.arange(size, dtype=np.int64)
    pc = _popcount(masks)
    weights = np.array(
        [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)] + [0.0]
    )
    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        phi[i] = float(np.sum(weights[pc[without]] * (values[without | bit] - values[without])))
    return phi


def shapley_exact(players: Sequence[str], v: Value) -> dict[str, float]:
    n = len(players)
    if n == 0:
        return {}
    if n > 20:
        raise ValueError("exact Shapley is limited to 20 players; use Monte-Carlo")
    table = np.empty(1 << n)
    for mask in range(1 << n):
        table[mask] = v(frozenset(players[i] for i in range(n) if mask >> i & 1))
    phi = shapley_from_table(table)
    return {p: float(phi[i]) for i, p in enumerate(players)}


@dataclass(frozen=True)
class MonteCarloShapley:
    values: dict[str, float]
    std_errors: dict[str, float]
    n_permutations: int


def shapley_monte_carlo(
    players: Sequence[str],
    v: Value,
    n_permutations: int,
    rng: np.random.Generator,
) -> MonteCarloShapley:
    """Permutation-sampling estimator; each sampled ordering is efficient by construction."""
    n = len(players)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    cache: dict[frozenset[str], float] = {}

    def cv(s: frozenset[str]) -> float:
        if s not in cache:
            cache[s] = v(s)
        return cache[s]

    samples = np.zeros((n_permutations, n))
    for t in range(n_permutations):
        order = rng.permutation(n)
        coalition: frozenset[str] = frozenset()
        prev = cv(coalition)
        for i in order:
            coalition = coalition | {players[i]}
            cur = cv(coalition)
            samples[t, i] = cur - prev
            prev = cur
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / np.sqrt(n_permutations) if n_permutations > 1 else mean * 0
    return MonteCarloShapley(
        values={p: float(mean[i]) for i, p in enumerate(players)},
        std_errors={p: float(se[i]) for i, p in enumerate(players)},
        n_permutations=n_permutations,
    )


def mc_convergence(
    players: Sequence[str],
    v: Value,
    sample_sizes: Sequence[int],
    rng: np.random.Generator,
    repeats: int = 5,
) -> list[tuple[int, float]]:
    """Mean absolute error of the MC estimate vs. exact, per number of permutations."""
    exact = shapley_exact(players, v)
    out = []
    for m in sample_sizes:
        errs = []
        for _ in range(repeats):
            est = shapley_monte_carlo(players, v, m, rng).values
            errs.append(float(np.mean([abs(est[p] - exact[p]) for p in players])))
        out.append((int(m), float(np.mean(errs))))
    return out


def _solve_game(
    players: Sequence[str], v: Value, exact_max: int, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    if len(players) <= exact_max:
        return shapley_exact(players, v)
    return shapley_monte_carlo(players, v, n_perm, rng).values


def _normalise_positive(phi: dict[str, float]) -> dict[str, float]:
    pos = {k: max(v, 0.0) for k, v in phi.items()}
    total = sum(pos.values())
    if total <= 0:
        return {k: 1.0 / len(phi) for k in phi}
    return {k: v / total for k, v in pos.items()}


# ------------------------------------------------------------------- empirical game


def empirical_rates(journeys: Sequence[Journey], prior_strength: float = 10.0) -> Value:
    """Smoothed conversion rate of journeys whose channel set is exactly S."""
    n_by: dict[frozenset[str], int] = {}
    c_by: dict[frozenset[str], int] = {}
    n_size: dict[int, int] = {}
    c_size: dict[int, int] = {}
    for j in journeys:
        if j.is_empty:
            continue
        s = frozenset(j.channels)
        n_by[s] = n_by.get(s, 0) + 1
        c_by[s] = c_by.get(s, 0) + int(j.converted)
        n_size[len(s)] = n_size.get(len(s), 0) + 1
        c_size[len(s)] = c_size.get(len(s), 0) + int(j.converted)
    total_n = sum(n_by.values())
    global_rate = sum(c_by.values()) / total_n if total_n else 0.0

    def v(s: frozenset[str]) -> float:
        if not s:
            return 0.0
        k = len(s)
        prior = c_size[k] / n_size[k] if n_size.get(k) else global_rate
        return (c_by.get(s, 0) + prior_strength * prior) / (n_by.get(s, 0) + prior_strength)

    return v


def shapley_empirical(
    journeys: Iterable[Journey],
    exact_max_channels: int = 10,
    n_permutations: int = 2000,
    seed: int = 0,
) -> Credits:
    js = list(journeys)
    v = empirical_rates(js)
    rng = np.random.default_rng(seed)
    cache: dict[frozenset[str], dict[str, float]] = {}
    credits: Credits = {}
    for j in attributable(js):
        key = frozenset(j.channels)
        if key not in cache:
            players = sorted(key)
            cache[key] = _normalise_positive(
                _solve_game(players, v, exact_max_channels, n_permutations, rng)
            )
        for ch, w in cache[key].items():
            credits[ch] = credits.get(ch, 0.0) + w
    return credits


# ------------------------------------------------------------------- model-based game


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -35, 35))), dtype=np.float64)


def fit_logistic(
    X: FloatArray, y: FloatArray, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8
) -> FloatArray:
    """Ridge logistic regression by IRLS (intercept in column 0 is not penalised)."""
    p = X.shape[1]
    beta = np.zeros(p)
    penalty = np.full(p, l2)
    penalty[0] = 0.0
    for _ in range(max_iter):
        mu = _sigmoid(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        grad = X.T @ (y - mu) - penalty * beta
        hess = (X * w[:, None]).T @ X + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


@dataclass(frozen=True)
class ResponseModel:
    channels: tuple[str, ...]
    coef: FloatArray  # [intercept, beta_channel...]

    def features(self, j: Journey) -> FloatArray:
        x = np.zeros(len(self.channels))
        idx = {c: i for i, c in enumerate(self.channels)}
        for c in j.channels:
            x[idx[c]] += 1.0
        return np.asarray(x, dtype=np.float64)

    def game(self, j: Journey) -> Value:
        x = self.features(j)
        idx = {c: i for i, c in enumerate(self.channels)}
        base = float(_sigmoid(np.array([self.coef[0]]))[0])

        def v(s: frozenset[str]) -> float:
            z = self.coef[0] + sum(self.coef[1 + idx[c]] * x[idx[c]] for c in s)
            return float(_sigmoid(np.array([z]))[0]) - base

        return v


def fit_response_model(journeys: Sequence[Journey], l2: float = 1.0) -> ResponseModel:
    js = [j for j in journeys if not j.is_empty]
    channels = tuple(sorted({c for j in js for c in j.channels}))
    proto = ResponseModel(channels=channels, coef=np.zeros(len(channels) + 1))
    X = np.column_stack([np.ones(len(js)), np.array([proto.features(j) for j in js])])
    y = np.array([float(j.converted) for j in js])
    return ResponseModel(channels=channels, coef=fit_logistic(X, y, l2=l2))


def shapley_model(
    journeys: Iterable[Journey],
    exact_max_channels: int = 10,
    n_permutations: int = 2000,
    seed: int = 0,
    l2: float = 1.0,
) -> Credits:
    js = list(journeys)
    if not attributable(js):
        return {}
    model = fit_response_model(js, l2=l2)
    rng = np.random.default_rng(seed)
    credits: Credits = {}
    cache: dict[tuple[float, ...], dict[str, float]] = {}
    for j in attributable(js):
        key = tuple(model.features(j).round(9))
        if key not in cache:
            players = sorted(set(j.channels))
            cache[key] = _normalise_positive(
                _solve_game(players, model.game(j), exact_max_channels, n_permutations, rng)
            )
        for ch, w in cache[key].items():
            credits[ch] = credits.get(ch, 0.0) + w
    return credits
