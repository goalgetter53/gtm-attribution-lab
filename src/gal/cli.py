"""Command-line interface: ``gal generate | build | attribute | validate | report``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from gal.attribution import run_all, to_shares
from gal.config import load_config, save_config
from gal.generator import load_ground_truth
from gal.pipeline import LEVELS, load_journeys
from gal.report import build_report
from gal.validation import run_validation, summarize
from gal.warehouse.build import DataTestError, build_warehouse, load_raw_parquet


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def cmd_generate(args: argparse.Namespace) -> int:
    from gal.generator import generate

    cfg = load_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.confounding is not None:
        overrides["confounding"] = args.confounding
    if args.n_accounts is not None:
        overrides["n_accounts"] = args.n_accounts
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    start = time.perf_counter()
    data = generate(cfg)
    data.write(args.out)
    save_config(cfg, Path(args.out) / "config.yaml")
    rows = {k: len(v) for k, v in data.tables.items()}
    print(f"generated SYNTHETIC data in {time.perf_counter() - start:.1f}s -> {args.out}")
    for k, v in rows.items():
        print(f"  raw.{k:<14} {v:>8,} rows")
    print(f"  realised wins: {data.ground_truth.realized_wins}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.raw) / "config.yaml")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db)
    try:
        load_raw_parquet(con, args.raw)
        try:
            result = build_warehouse(con, cfg, args.models)
        except DataTestError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    finally:
        con.close()
    for name in result.executed:
        print(f"  model {name:<28} {result.seconds[name] * 1000:7.1f} ms")
    passed = sum(t.passed for t in result.tests)
    print(f"built {len(result.executed)} models; data tests {passed}/{len(result.tests)} passed")
    return 0


def attribution_frame(con: duckdb.DuckDBPyConnection, raw_dir: str | Path) -> pd.DataFrame:
    cfg = load_config(Path(raw_dir) / "config.yaml")
    gt = load_ground_truth(Path(raw_dir) / "ground_truth.json")
    rows = []
    for level in LEVELS:
        credits_by_model = run_all(load_journeys(con, level), cfg.attribution, seed=cfg.seed)
        for model, credits in credits_by_model.items():
            shares = to_shares(credits, list(gt.channels))
            for ch in sorted(set(credits) | set(gt.channels)):
                rows.append(
                    {
                        "level": level,
                        "model": model,
                        "channel": ch,
                        "credit": float(credits.get(ch, 0.0)),
                        "share": shares.get(ch, float("nan")),
                        "truth_share": gt.share.get(ch, float("nan")),
                    }
                )
    return pd.DataFrame(rows)


def cmd_attribute(args: argparse.Namespace) -> int:
    con = duckdb.connect(args.db, read_only=True)
    try:
        df = attribution_frame(con, args.raw)
    finally:
        con.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    acc = df[(df.level == "account") & df.truth_share.notna()]
    pivot = acc.pivot(index="model", columns="channel", values="share").round(3)
    truth = acc.drop_duplicates("channel").set_index("channel")["truth_share"].round(3)
    pivot.loc["GROUND_TRUTH"] = truth
    print(pivot.to_string())
    print(f"wrote {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seeds = _ints(args.seeds) if args.seeds else list(range(args.n_seeds))
    levels = _floats(args.confounding) if args.confounding else [cfg.confounding]
    if args.n_accounts:
        cfg = cfg.with_overrides(n_accounts=args.n_accounts)
    start = time.perf_counter()
    runs = run_validation(cfg, seeds, levels, args.models, progress=lambda m: print(f"  run {m}"))
    summary = summarize(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    elapsed = time.perf_counter() - start
    print(f"{len(seeds) * len(levels)} synthetic runs in {elapsed:.0f}s -> {out}")
    acc = summary[summary.level == "account"]
    cols = ["confounding", "model", "mae_mean", "mae_ci_low", "mae_ci_high", "spearman_mean"]
    print(acc[cols].round(3).to_string(index=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.raw) / "config.yaml")
    gt = load_ground_truth(Path(args.raw) / "ground_truth.json")
    con = duckdb.connect(args.db, read_only=True)
    try:
        attribution = (
            pd.read_csv(args.attribution) if args.attribution else attribution_frame(con, args.raw)
        )
        summary = pd.read_csv(Path(args.validation) / "summary.csv") if args.validation else None
        control = pd.read_csv(Path(args.control) / "summary.csv") if args.control else None
        rep = build_report(
            con,
            gt,
            attribution,
            cfg.data_end_day,
            load_journeys(con, "account"),
            summary,
            control,
            reference_confounding=cfg.confounding,
        )
    finally:
        con.close()
    html_path, md_path = rep.write(args.out)
    print(f"wrote {html_path} and {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gal", description="GTM attribution lab (synthetic data)")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic raw dataset")
    g.add_argument("-c", "--config", default="configs/default.yaml")
    g.add_argument("-o", "--out", default="data/raw")
    g.add_argument("--seed", type=int)
    g.add_argument("--confounding", type=float)
    g.add_argument("--n-accounts", type=int)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("build", help="load raw data into DuckDB and run models + tests")
    b.add_argument("--raw", default="data/raw")
    b.add_argument("--db", default="data/warehouse.duckdb")
    b.add_argument("--models", default=None, help="models directory (default: repo models/)")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("attribute", help="run every attribution model")
    a.add_argument("--raw", default="data/raw")
    a.add_argument("--db", default="data/warehouse.duckdb")
    a.add_argument("-o", "--out", default="results/attribution.csv")
    a.set_defaults(func=cmd_attribute)

    v = sub.add_parser("validate", help="score models vs ground truth across seeds/confounding")
    v.add_argument("-c", "--config", default="configs/default.yaml")
    v.add_argument("--seeds", help="comma-separated seeds (overrides --n-seeds)")
    v.add_argument("--n-seeds", type=int, default=5)
    v.add_argument("--confounding", help="comma-separated confounding levels")
    v.add_argument("--n-accounts", type=int)
    v.add_argument("--models", default=None)
    v.add_argument("-o", "--out", default="results/validation")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("report", help="write the static HTML + Markdown report")
    r.add_argument("--raw", default="data/raw")
    r.add_argument("--db", default="data/warehouse.duckdb")
    r.add_argument("--attribution", default=None)
    r.add_argument("--validation", default=None, help="directory with summary.csv")
    r.add_argument("--control", default=None, help="positive-control directory with summary.csv")
    r.add_argument("-o", "--out", default="reports")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
