"""Score every attribution model against the counterfactual ground truth.

For each (seed, confounding level) we generate a fresh synthetic world, build the
warehouse, run every model at account and contact level, and compare the model's channel
credit *shares* with the ground-truth incremental-win shares:

* MAE   = mean over channels of |model share - true share|  (lower is better)
* rho   = Spearman rank correlation of shares                (higher is better)

Credit landing on 'direct' (untracked traffic) has no ground truth; it is excluded and
reported separately as ``direct_share``. A ``uniform`` reference model (equal shares)
anchors the scale. Intervals are 95% t-intervals across seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from gal.attribution import Credits, to_shares
from gal.config import SimConfig
from gal.generator import GroundTruth
from gal.pipeline import LEVELS, run_pipeline
from gal.stats import mean_abs_error, spearman, t_interval


def score(credits: Credits, gt: GroundTruth) -> dict[str, float]:
    channels = list(gt.channels)
    shares = to_shares(credits, channels)
    truth = gt.share
    total = sum(max(v, 0.0) for v in credits.values())
    direct = max(credits.get("direct", 0.0), 0.0) / total if total > 0 else 0.0
    return {
        "mae": mean_abs_error([shares[c] for c in channels], [truth[c] for c in channels]),
        "spearman": spearman([shares[c] for c in channels], [truth[c] for c in channels]),
        "direct_share": direct,
    }


def evaluate_run(cfg: SimConfig, models_dir: str | Path | None = None) -> pd.DataFrame:
    run = run_pipeline(cfg, models_dir)
    gt = run.data.ground_truth
    rows = []
    for level in LEVELS:
        model_credits = dict(run.credits[level])
        model_credits["uniform"] = dict.fromkeys(gt.channels, 1.0)
        for model, credits in model_credits.items():
            s = score(credits, gt)
            shares = to_shares(credits, list(gt.channels))
            rows.append(
                {
                    "seed": cfg.seed,
                    "confounding": cfg.confounding,
                    "level": level,
                    "model": model,
                    **s,
                    "realized_wins": gt.realized_wins,
                    **{f"share_{c}": shares[c] for c in gt.channels},
                    **{f"truth_{c}": gt.share[c] for c in gt.channels},
                }
            )
    run.con.close()
    return pd.DataFrame(rows)


def run_validation(
    cfg: SimConfig,
    seeds: Sequence[int],
    confounding_levels: Sequence[float],
    models_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    frames = []
    for conf in confounding_levels:
        for seed in seeds:
            if progress:
                progress(f"confounding={conf} seed={seed}")
            frames.append(evaluate_run(cfg.with_overrides(seed=seed, confounding=conf), models_dir))
    return pd.concat(frames, ignore_index=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% t-interval of MAE and Spearman per (confounding, level, model)."""
    rows = []
    for (conf, level, model), g in runs.groupby(["confounding", "level", "model"], sort=True):
        mae = t_interval(g["mae"].tolist())
        rho = t_interval(g["spearman"].tolist())
        rows.append(
            {
                "confounding": conf,
                "level": level,
                "model": model,
                "n_seeds": mae.n,
                "mae_mean": mae.mean,
                "mae_ci_low": mae.low,
                "mae_ci_high": mae.high,
                "spearman_mean": rho.mean,
                "spearman_ci_low": rho.low,
                "spearman_ci_high": rho.high,
                "direct_share_mean": float(g["direct_share"].mean()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["confounding", "level", "mae_mean"]).reset_index(drop=True)
    )
