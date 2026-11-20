"""Run the lab's attribution models on the real Criteo attribution dataset.

Usage (after ``make real-criteo-data``)::

    python scripts/run_criteo.py --parquet data/real/criteo.parquet

Every number in ``reports/real_criteo.md`` is written by this script from its own results.
There is no ground truth on real data: the outputs measure how much the models disagree,
and how they compare with Criteo's own last-click label.
"""

from __future__ import annotations

import os

# Bootstrap workers are forked; a multi-threaded BLAS can deadlock in a forked child.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gal.attribution import heuristics, markov, model_registry, shapley, to_shares
from gal.attribution.base import Credits, Journey
from gal.config import AttributionConfig
from gal.real import criteo
from gal.real.compare import gap, percentile_ci, shares_by_model, spearman_matrix

SKIPPED_MODELS = {"w_shaped": "no MQL/opportunity milestones in display-ad data"}
BOOT_MODELS = ("last_touch", "markov_o1", "shapley_empirical")
PAIRS = (
    ("last_touch", "markov_o1"),
    ("last_touch", "markov_o2"),
    ("last_touch", "shapley_empirical"),
    ("last_touch", "shapley_model"),
    ("markov_o1", "shapley_empirical"),
    ("first_touch", "last_touch"),
    ("criteo_label", "last_touch"),
    ("criteo_label", "markov_o1"),
    ("criteo_label", "shapley_empirical"),
)
BOOT_PAIRS = (
    ("last_touch", "markov_o1"),
    ("last_touch", "shapley_empirical"),
    ("markov_o1", "shapley_empirical"),
    ("criteo_label", "last_touch"),
    ("criteo_label", "markov_o1"),
)

_STATE: dict[str, Any] = {}


def connect(parquet: Path, tmp: Path) -> duckdb.DuckDBPyConnection:
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='3GB'; SET threads=2; SET temp_directory='{tmp.as_posix()}'")
    criteo.register_source(con, parquet)
    return con


def run_models(
    js: Sequence[Journey], cfg: AttributionConfig
) -> tuple[dict[str, Credits], dict[str, float]]:
    out: dict[str, Credits] = {}
    secs: dict[str, float] = {}
    for name, fn in model_registry(cfg).items():
        if name in SKIPPED_MODELS:
            continue
        t = time.time()
        out[name] = fn(js)
        secs[name] = round(time.time() - t, 1)
        print(f"  {name:<18} {secs[name]:>6.1f}s", flush=True)
    return out, secs


def _replicate(seed: int) -> dict[str, dict[str, float]]:
    cj: criteo.CriteoJourneys = _STATE["cj"]
    exact_max: int = _STATE["exact_max"]
    rng = np.random.default_rng(seed)
    conv_counts, path_counts = cj.bootstrap_counts(rng)
    js = cj.expand(conv_counts, path_counts)
    credits: dict[str, Credits] = {
        "last_touch": heuristics.last_touch(js),
        "markov_o1": markov.markov_attribution(js, order=1),
        "shapley_empirical": shapley.shapley_empirical(js, exact_max),
        "criteo_label": criteo.weighted_label_counts(cj.criteo_channel, conv_counts, cj.channels),
    }
    return {m: to_shares(c, cj.channels) for m, c in credits.items()}


def bootstrap(
    cj: criteo.CriteoJourneys, n_boot: int, workers: int, seed: int, exact_max: int
) -> list[dict[str, dict[str, float]]]:
    _STATE.update(cj=cj, exact_max=exact_max)
    seeds = [seed + 1000 + b for b in range(n_boot)]
    if workers <= 1:
        return [_replicate(s) for s in seeds]
    with mp.get_context("fork").Pool(workers) as pool:
        reps = []
        for i, r in enumerate(pool.imap(_replicate, seeds), 1):
            reps.append(r)
            if i % 5 == 0:
                print(f"  bootstrap {i}/{n_boot}", flush=True)
    return reps


def boot_summary(
    reps: list[dict[str, dict[str, float]]], channels: Sequence[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_replicates": len(reps), "pairs": {}}
    for a, b in BOOT_PAIRS:
        gs = [gap(a, r[a], b, r[b], channels) for r in reps]
        per_channel = {c: percentile_ci([r[a][c] - r[b][c] for r in reps]) for c in channels}
        named = [c for c in channels if c != criteo.OTHER]
        gn = [gap(a, r[a], b, r[b], named) for r in reps]
        out["pairs"][f"{a}__vs__{b}"] = {
            "max_abs_gap_ci": percentile_ci([g.max_abs_gap for g in gs]),
            "max_abs_gap_named_ci": percentile_ci([g.max_abs_gap for g in gn]),
            "spearman_ci": percentile_ci([g.spearman for g in gs]),
            "total_variation_ci": percentile_ci([g.total_variation for g in gs]),
            "per_channel_gap_ci": per_channel,
        }
    return out


def chart(
    shares: dict[str, dict[str, float]], channels: Sequence[str], n_other: int, path: Path
) -> None:
    models = ["first_touch", "last_touch", "markov_o1", "shapley_empirical", "criteo_label"]
    colors = ["#9aa5b1", "#d1495b", "#2e86ab", "#3f8f5a", "#edae49"]
    camp = [c for c in channels if c != criteo.OTHER]
    y = np.arange(len(camp))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)
    for i, (m, col) in enumerate(zip(models, colors, strict=True)):
        vals = [100 * shares[m][c] for c in camp]
        ax.barh(y + (i - (len(models) - 1) / 2) * h, vals, height=h, color=col, label=m)
    ax.set_yticks(y, camp)
    ax.invert_yaxis()
    ax.set_xlabel("share of conversion credit (%)")
    other = ", ".join(f"{m} {100 * shares[m][criteo.OTHER]:.1f}%" for m in models)
    ax.set_title(
        f"Criteo (real data): credit share per top-{len(camp)} campaign, by model\n"
        f"'other' ({n_other} smaller campaigns) not drawn:\n{other}",
        fontsize=9,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- markdown


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def pp(x: float, nd: int = 2) -> str:
    return f"{100 * x:+.{nd}f} pp"


def ci_pp(ci: Sequence[float], nd: int = 2) -> str:
    return f"[{100 * ci[0]:.{nd}f}, {100 * ci[1]:.{nd}f}]"


def fmt_int(x: float) -> str:
    return f"{round(x):,}"


def render_md(res: dict[str, Any]) -> str:
    s = res["stats"]
    ch = res["channels"]
    sh = res["shares"]
    models = res["models"]
    g = {f"{p['a']}__vs__{p['b']}": p for p in res["gaps"]}
    boot = res.get("bootstrap") or {"n_replicates": 0, "pairs": {}}
    bp = boot["pairs"]
    L: list[str] = []
    w = L.append
    w("# Attribution models on real data: Criteo (16.5M impressions)")
    w("")
    w(
        "> **Real, public data. No ground truth.** Source: Criteo Attribution Modeling for "
        "Bidding dataset (Diemert et al. 2017, "
        "[arXiv:1707.06409](https://arxiv.org/abs/1707.06409)), "
        "[Hugging Face](https://huggingface.co/datasets/criteo/criteo-attribution-dataset), "
        "licence CC BY-NC-SA 4.0. Every number here is written by `scripts/run_criteo.py`."
    )
    w("")
    w("## Data and journeys")
    w("")
    w(
        f"- Impressions: **{fmt_int(s['impressions'])}**; users: **{fmt_int(s['users'])}**; "
        f"campaigns: **{fmt_int(s['campaigns'])}**; clicks: {fmt_int(s['clicks'])}."
    )
    w(
        f"- Conversion events (distinct user × conversion_id): **{fmt_int(s['conversions'])}** "
        f"({fmt_int(s['conversion_ids'])} distinct conversion_ids). "
        f"Criteo's own `attribution` label credits {fmt_int(s['conversions_criteo_attributed'])} "
        f"of them ({pct(s['conversions_criteo_attributed'] / s['conversions'], 1)}) to Criteo."
    )
    w(
        f"- Journeys: each user's impressions are cut at every conversion (ASOF join on time). "
        f"{fmt_int(s['conversions'])} converting journeys (mean "
        f"{s['mean_touches_converted']:.2f} touches; {fmt_int(s['converted_journeys_empty'])} "
        f"are empty because a same-timestamp conversion took the touches) and "
        f"{fmt_int(s['nonconverting_journeys'])} non-converting journeys "
        f"({fmt_int(s['unique_nonconverting_paths'])} unique channel paths)."
    )
    w(
        f"- Channels: the top **K = {int(s['top_k'])}** campaigns by impressions (`c01` = largest) "
        f"plus `other` for the remaining {fmt_int(s['campaigns'] - s['top_k'])}. The top "
        f"{int(s['top_k'])} hold {pct(s['top_k_impression_share'], 1)} of impressions. "
        "A small K keeps exact Shapley values and a dense order-2 Markov chain tractable."
    )
    w(
        f"- Only **{fmt_int(s['converted_journeys_multi_channel'])}** converting journeys "
        f"({pct(s['converted_journeys_multi_channel'] / s['conversions'], 1)}) touch more than one "
        "channel. On the rest every model must agree, so all disagreement below comes from that "
        "multi-channel minority."
    )
    w(
        f"- Criteo's conversion labels are campaign-specific: only "
        f"{fmt_int(s['conversion_ids_multi_campaign'])} of {fmt_int(s['conversion_ids'])} "
        f"conversion_ids ({pct(s['conversion_ids_multi_campaign'] / s['conversion_ids'], 2)}) "
        "carry impressions from more than one campaign. A user-level journey can therefore mix "
        "campaigns (likely different advertisers) that never shared a conversion label. "
        f"In {pct(res['diagnostics']['last_touch_not_owner_share'], 1)} of non-empty converting "
        "journeys the last touch is a different channel from the campaign that owns the "
        "conversion_id."
    )
    w(
        f"- Settings: time-decay half-life {res['settings']['half_life_days']:g} days; "
        "Markov order 1 and 2; Shapley exact over each journey's channel set; no horizon "
        "matching (the dataset has no lead-creation time). W-shaped is skipped: there are "
        "no MQL/opportunity milestones in display-ad data."
    )
    w("")
    w("## Credit share per channel (% of all conversion credit)")
    w("")
    cols = [*models, "criteo_label"]
    w("| channel | campaign id | " + " | ".join(cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    for c in ch:
        cid = res["campaign_ids"].get(c, f"{fmt_int(s['campaigns'] - s['top_k'])} others")
        w(f"| {c} | {cid} | " + " | ".join(pct(sh[m][c]) for m in cols) + " |")
    w("")
    w(
        "`criteo_label` = share of Criteo-attributed conversions whose last attributed click was "
        "on that channel (Criteo's own last-click attribution). Model shares are over all "
        "conversions with at least one touch."
    )
    w("")
    w("![credit shares](figures/real_criteo_shares.png)")
    w("")
    w("## How far the models disagree")
    w("")
    w(
        "Gap = share(a) − share(b) in percentage points. Total variation = share of all credit "
        "that would have to move to turn one model's split into the other's. Spearman ρ is "
        f"over the {len(ch)} channel shares."
    )
    w("")
    w(
        f"| a vs b | max abs gap | on channel | max abs gap, top-{int(s['top_k'])} only | "
        "a over-credits most | "
        "a under-credits most | total variation | Spearman ρ |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for p in res["gaps"]:
        w(
            f"| {p['a']} vs {p['b']} | {pp(p['max_abs_gap'])} | {p['max_abs_gap_channel']} | "
            f"{pp(p['max_abs_gap_named'])} ({p['max_abs_gap_named_channel']}) | "
            f"{p['most_over']} ({pp(p['most_over_gap'])}) | "
            f"{p['most_under']} ({pp(p['most_under_gap'])}) | {pct(p['total_variation'])} | "
            f"{p['spearman']:.3f} |"
        )
    w("")
    if bp:
        w(f"### Bootstrap 95% intervals ({boot['n_replicates']} Poisson replicates over users)")
        w("")
        w(
            f"| a vs b | max abs gap [95% CI] | max abs gap, top-{int(s['top_k'])} only [95% CI] | "
            "total variation [95% CI] | Spearman ρ [95% CI] |"
        )
        w("|---|---|---|---|---|")
        for key, v in bp.items():
            a, b = key.split("__vs__")
            pt = g.get(key)
            if pt is None:
                continue
            w(
                f"| {a} vs {b} | {pp(pt['max_abs_gap'])} {ci_pp(v['max_abs_gap_ci'])} | "
                f"{pp(pt['max_abs_gap_named'])} {ci_pp(v['max_abs_gap_named_ci'])} | "
                f"{pct(pt['total_variation'])} {ci_pp(v['total_variation_ci'])} | "
                f"{pt['spearman']:.3f} [{v['spearman_ci'][0]:.3f}, {v['spearman_ci'][1]:.3f}] |"
            )
        w("")
        w(
            "Per-channel gap, last touch minus Markov order 1 and minus empirical Shapley "
            "(pp, point [95% CI]):"
        )
        w("")
        w("| channel | last − markov_o1 | last − shapley_empirical | last / markov_o1 credit |")
        w("|---|---|---|---|")
        lm = bp["last_touch__vs__markov_o1"]["per_channel_gap_ci"]
        ls = bp["last_touch__vs__shapley_empirical"]["per_channel_gap_ci"]
        for c in ch:
            d1 = sh["last_touch"][c] - sh["markov_o1"][c]
            d2 = sh["last_touch"][c] - sh["shapley_empirical"][c]
            ratio = sh["last_touch"][c] / sh["markov_o1"][c] if sh["markov_o1"][c] > 0 else np.nan
            w(f"| {c} | {pp(d1)} {ci_pp(lm[c])} | {pp(d2)} {ci_pp(ls[c])} | {ratio:.2f}× |")
        w("")
    w("## Spearman ρ between models (channel shares)")
    w("")
    sm = res["spearman_matrix"]
    names = list(sm)
    w("| | " + " | ".join(names) + " |")
    w("|---|" + "---|" * len(names))
    for a in names:
        w(f"| {a} | " + " | ".join(f"{sm[a][b]:.2f}" for b in names) + " |")
    w("")
    tt = res.get("touch_type")
    if tt:
        w("## Variant: channel = click vs view")
        w("")
        w(
            "Same journeys, but every impression is labelled `click` or `view` instead of by "
            "campaign. Share of conversion credit given to clicked impressions:"
        )
        w("")
        w("| model | click share | view share |")
        w("|---|---|---|")
        for m, v in tt["shares"].items():
            w(f"| {m} | {pct(v['click'], 1)} | {pct(v['view'], 1)} |")
        w("")
        w(
            f"{fmt_int(s['conversion_ids_with_click'])} of {fmt_int(s['conversion_ids'])} "
            "conversion_ids have at least one clicked impression (the dataset tracks "
            "post-click conversions), so click-based rules have a built-in advantage here."
        )
        w("")
    w("## What this does and does not show")
    w("")
    w(
        "- There is no causal ground truth in this dataset. These numbers show how far the "
        "models disagree on real journeys, and how close each is to Criteo's own last-click "
        "label. They cannot say which model is right. The synthetic lab in this repo is what "
        "scores models against a known truth."
    )
    w(
        f"- `other` bundles {fmt_int(s['campaigns'] - s['top_k'])} campaigns and dominates every "
        f"model's split, so the top-{int(s['top_k'])} "
        "campaign shares are small in absolute terms; relative differences (last/markov ratio) "
        "are the clearer signal."
    )
    w(
        "- Journeys cross campaigns that probably belong to different advertisers. Crediting a "
        "conversion to another advertiser's campaign is a modelling artefact of user-level "
        "journeys, not evidence that the campaign caused it."
    )
    w(
        "- Non-converting journeys are right-censored at the end of the 30-day window and are "
        "not horizon-matched."
    )
    w(
        "- Bootstrap intervals resample users (Poisson weights) and hold the top-K channel "
        "set fixed. They cover sampling noise, not model or definition choices."
    )
    w("")
    w(
        f"Runtime: {res['runtime_seconds']:.0f} s ({res['settings']['bootstrap_replicates']} "
        "bootstrap replicates)."
    )
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=Path("data/real/criteo.parquet"))
    ap.add_argument("--tsv", type=Path, default=Path("data/real/criteo_attribution_dataset.tsv.gz"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--half-life", type=float, default=7.0)
    ap.add_argument("--boot", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-touch-type", action="store_true")
    ap.add_argument(
        "--from-json",
        action="store_true",
        help="re-render the markdown and chart from an existing --json result (no model runs)",
    )
    ap.add_argument("--csv", type=Path, default=Path("results/real_criteo_attribution.csv"))
    ap.add_argument("--json", type=Path, default=Path("reports/real_criteo.json"))
    ap.add_argument("--md", type=Path, default=Path("reports/real_criteo.md"))
    ap.add_argument("--fig", type=Path, default=Path("reports/figures/real_criteo_shares.png"))
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.from_json:
        res: dict[str, Any] = json.loads(args.json.read_text())
        n_other = int(res["stats"]["campaigns"] - res["stats"]["top_k"])
        chart(res["shares"], res["channels"], n_other, args.fig)
        args.md.write_text(render_md(res))
        print(f"re-rendered {args.md} and {args.fig} from {args.json}")
        return 0

    if not args.parquet.exists():
        if not args.tsv.exists():
            raise SystemExit(f"missing {args.tsv}; download it from {criteo.CRITEO_URL}")
        print(f"converting {args.tsv} -> {args.parquet}", flush=True)
        criteo.tsv_to_parquet(args.tsv, args.parquet)

    exact_max = args.top_k + 1  # every channel incl. 'other': Shapley is always exact
    cfg = AttributionConfig(
        window_days=30,
        half_life_days=args.half_life,
        markov_order=2,
        shapley_exact_max_channels=exact_max,
        shapley_mc_permutations=2000,
        session_timeout_minutes=30,
    )

    print("building campaign journeys", flush=True)
    con = connect(args.parquet, args.parquet.parent / "tmp")
    cj = criteo.load_journeys(con, top_k=args.top_k, channel_mode="campaign")
    campaign_ids = {
        str(lab): int(cid)
        for lab, cid in con.execute("SELECT label, campaign FROM topk ORDER BY rnk").fetchall()
    }
    con.close()
    channels = cj.channels

    # Bootstrap first, while this process is small (workers are forked copies of it).
    boot = None
    if args.boot > 0:
        print(f"bootstrap: {args.boot} replicates on {args.workers} workers", flush=True)
        reps = bootstrap(cj, args.boot, args.workers, args.seed, exact_max)
        boot = boot_summary(reps, channels)

    js = cj.expand()
    print(f"  {len(js):,} journeys, {cj.stats['conversions']:,.0f} conversions", flush=True)

    credits, secs = run_models(js, cfg)
    credits["criteo_label"] = criteo.label_counts(cj.criteo_channel, channels)
    credits["conversion_owner"] = criteo.label_counts(cj.owner_channel, channels)
    del js
    shares = shares_by_model(credits, channels)
    models = [m for m in credits if m not in ("criteo_label", "conversion_owner")]

    nonempty = [(j, o) for j, o in zip(cj.converted, cj.owner_channel, strict=True) if j.channels]
    diagnostics = {
        "last_touch_not_owner_share": sum(j.channels[-1] != o for j, o in nonempty) / len(nonempty),
        "nonempty_conversions": len(nonempty),
    }
    del nonempty

    named = [c for c in channels if c != criteo.OTHER]
    gaps = []
    for a, b in PAIRS:
        g_all = gap(a, shares[a], b, shares[b], channels).__dict__
        g_named = gap(a, shares[a], b, shares[b], named)
        g_all["max_abs_gap_named"] = g_named.max_abs_gap
        g_all["max_abs_gap_named_channel"] = g_named.max_abs_gap_channel
        gaps.append(g_all)
    smat = spearman_matrix({m: shares[m] for m in [*models, "criteo_label"]}, channels)

    rows = [
        {
            "variant": "campaign",
            "model": m,
            "channel": c,
            "credit": credits[m].get(c, 0.0),
            "share": shares[m][c],
        }
        for m in credits
        for c in channels
    ]
    stats = cj.stats
    del cj

    touch_type = None
    if not args.no_touch_type:
        print("building click/view journeys", flush=True)
        con = connect(args.parquet, args.parquet.parent / "tmp")
        tj = criteo.load_journeys(con, top_k=args.top_k, channel_mode="touch_type")
        con.close()
        tcred, _ = run_models(tj.expand(), cfg)
        tshares = shares_by_model(tcred, tj.channels)
        touch_type = {"shares": tshares}
        rows += [
            {
                "variant": "touch_type",
                "model": m,
                "channel": c,
                "credit": tcred[m].get(c, 0.0),
                "share": tshares[m][c],
            }
            for m in tcred
            for c in tj.channels
        ]
        del tj

    res = {
        "source": {
            "name": "Criteo Attribution Modeling for Bidding dataset",
            "url": criteo.CRITEO_URL,
            "license": "CC BY-NC-SA 4.0",
            "citation": "Diemert, Meynet, Galland, Lefortier (2017). Attribution Modeling "
            "Increases Efficiency of Bidding in Display Advertising. AdKDD & TargetAd "
            "(KDD 2017). arXiv:1707.06409",
        },
        "settings": {
            "top_k": args.top_k,
            "half_life_days": args.half_life,
            "markov_orders": [1, 2],
            "shapley_exact_max_channels": exact_max,
            "bootstrap_replicates": args.boot,
            "seed": args.seed,
            "skipped_models": SKIPPED_MODELS,
        },
        "stats": stats,
        "channels": list(channels),
        "campaign_ids": campaign_ids,
        "models": models,
        "model_seconds": secs,
        "credits": credits,
        "shares": shares,
        "gaps": gaps,
        "spearman_matrix": smat,
        "diagnostics": diagnostics,
        "bootstrap": boot,
        "touch_type": touch_type,
    }
    res["runtime_seconds"] = round(time.time() - t0, 1)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False, float_format="%.6g")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(res, indent=2, default=float))
    n_other = int(stats["campaigns"] - stats["top_k"])
    chart(shares, channels, n_other, args.fig)
    args.md.write_text(render_md(res))
    print(f"wrote {args.csv}, {args.json}, {args.md}, {args.fig} in {res['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
