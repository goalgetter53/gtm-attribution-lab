from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from gal.cli import main
from gal.config import SimConfig
from gal.generator import GroundTruth
from gal.pipeline import PipelineRun
from gal.validation import evaluate_run, score, summarize


def _gt() -> GroundTruth:
    return GroundTruth(("a", "b"), 10.0, {"a": 3.0, "b": 1.0}, 9, 4, 0.0, 1)


def test_score_perfect_and_direct_share() -> None:
    s = score({"a": 6.0, "b": 2.0, "direct": 2.0}, _gt())
    assert s["mae"] == pytest.approx(0.0)
    assert s["direct_share"] == pytest.approx(0.2)


def test_score_wrong_ranking() -> None:
    s = score({"a": 1.0, "b": 3.0}, _gt())
    assert s["mae"] == pytest.approx(0.5)
    assert s["spearman"] == pytest.approx(-1.0)


def test_evaluate_and_summarize(small_cfg: SimConfig) -> None:
    runs = pd.concat(
        [evaluate_run(small_cfg.with_overrides(seed=s, n_accounts=250)) for s in (1, 2)]
    )
    assert {"uniform", "last_touch", "markov_o1", "shapley_model"} <= set(runs.model)
    assert set(runs.level) == {"account", "contact"}
    summary = summarize(runs)
    assert (summary.n_seeds == 2).all()
    assert (summary.mae_ci_low <= summary.mae_mean).all()
    assert (summary.mae_mean <= summary.mae_ci_high).all()


def test_pipeline_credits_sum_to_attributable(small_run: PipelineRun) -> None:
    from gal.attribution.base import total_attributable

    for level, by_model in small_run.credits.items():
        n = total_attributable(small_run.journeys[level])
        for model, credits in by_model.items():
            assert sum(credits.values()) == pytest.approx(n), (level, model)


def test_contact_level_sees_fewer_touches(small_run: PipelineRun) -> None:
    acc = sum(len(j.channels) for j in small_run.journeys["account"])
    con = sum(len(j.channels) for j in small_run.journeys["contact"])
    assert 0 < con < acc


@pytest.mark.slow
def test_cli_end_to_end(tmp_path: Path, default_dict: dict[str, Any]) -> None:
    d = dict(default_dict)
    d.update(n_accounts=250, gt_replicates=4)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(d))
    raw, db = tmp_path / "raw", tmp_path / "wh.duckdb"
    assert main(["generate", "-c", str(cfg_path), "-o", str(raw), "--seed", "3"]) == 0
    assert main(["build", "--raw", str(raw), "--db", str(db)]) == 0
    attr = tmp_path / "attr.csv"
    assert main(["attribute", "--raw", str(raw), "--db", str(db), "-o", str(attr)]) == 0
    df = pd.read_csv(attr)
    assert {"level", "model", "channel", "credit", "share", "truth_share"} <= set(df.columns)
    val = tmp_path / "val"
    assert (
        main(
            [
                "validate",
                "-c",
                str(cfg_path),
                "--seeds",
                "1,2",
                "--confounding",
                "0,1",
                "--n-accounts",
                "200",
                "-o",
                str(val),
            ]
        )
        == 0
    )
    assert (val / "summary.csv").exists() and (val / "runs.csv").exists()
    out = tmp_path / "rep"
    assert (
        main(
            [
                "report",
                "--raw",
                str(raw),
                "--db",
                str(db),
                "--attribution",
                str(attr),
                "--validation",
                str(val),
                "--control",
                str(val),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    html = (out / "report.html").read_text()
    assert "data:image/png;base64," in html and "SYNTHETIC" in html
    md = (out / "report.md").read_text()
    assert "figures/mae_by_confounding.png" in md
    assert (out / "figures" / "km_time_to_stage.png").stat().st_size > 1000


def test_build_fails_cleanly_without_raw(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["build", "--raw", str(tmp_path / "nope"), "--db", str(tmp_path / "x.duckdb")])
