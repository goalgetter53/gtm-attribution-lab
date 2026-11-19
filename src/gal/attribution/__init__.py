"""Attribution models: heuristics, Markov removal effect and Shapley values."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from gal.attribution import heuristics, markov, shapley
from gal.attribution.base import Credits, Journey, attributable, match_horizons, to_shares
from gal.config import AttributionConfig

ModelFn = Callable[[Sequence[Journey]], Credits]

DATA_DRIVEN_PREFIXES: tuple[str, ...] = ("markov", "shapley")
"""Models that contrast converting with non-converting journeys (get matched horizons)."""


def model_registry(cfg: AttributionConfig) -> dict[str, ModelFn]:
    """Every model evaluated by the lab, keyed by a stable name."""
    reg: dict[str, ModelFn] = {
        "first_touch": heuristics.first_touch,
        "last_touch": heuristics.last_touch,
        "linear": heuristics.linear,
        "time_decay": lambda js: heuristics.time_decay(js, cfg.half_life_days),
        "u_shaped": heuristics.u_shaped,
        "w_shaped": heuristics.w_shaped,
    }
    for k in range(1, cfg.markov_order + 1):
        reg[f"markov_o{k}"] = lambda js, k=k: markov.markov_attribution(js, order=k)  # type: ignore[misc]
    reg["shapley_empirical"] = lambda js: shapley.shapley_empirical(
        js, cfg.shapley_exact_max_channels, cfg.shapley_mc_permutations
    )
    reg["shapley_model"] = lambda js: shapley.shapley_model(
        js, cfg.shapley_exact_max_channels, cfg.shapley_mc_permutations
    )
    return reg


def run_all(
    journeys: Sequence[Journey], cfg: AttributionConfig, seed: int = 0
) -> dict[str, Credits]:
    """Run every model. Data-driven models see horizon-matched non-converting journeys."""
    matched = match_horizons(journeys, np.random.default_rng(seed))
    out: dict[str, Credits] = {}
    for name, fn in model_registry(cfg).items():
        out[name] = fn(matched if name.startswith(DATA_DRIVEN_PREFIXES) else journeys)
    return out


__all__ = [
    "Credits",
    "Journey",
    "attributable",
    "match_horizons",
    "model_registry",
    "run_all",
    "to_shares",
]
