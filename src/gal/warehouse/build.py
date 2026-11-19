"""Load raw tables into DuckDB and build the layered models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gal.config import SimConfig
from gal.generator import EPOCH, RAW_TABLES
from gal.warehouse.runner import RunResult, run_models, run_tests


class DataTestError(RuntimeError):
    """Raised when one or more warehouse data tests fail."""


def default_models_dir() -> Path:
    """``models/`` at the repository root (this package is used from a source checkout)."""
    return Path(__file__).resolve().parents[3] / "models"


def warehouse_vars(cfg: SimConfig) -> dict[str, Any]:
    data_end = EPOCH + pd.Timedelta(days=cfg.data_end_day)
    return {
        "data_end_ts": data_end.strftime("%Y-%m-%d %H:%M:%S"),
        "attribution_window_days": cfg.attribution.window_days,
        "session_timeout_minutes": cfg.attribution.session_timeout_minutes,
    }


def load_raw_frames(con: Any, tables: dict[str, pd.DataFrame]) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    for name in RAW_TABLES:
        df = tables[name]
        con.register("_raw_df", df)
        con.execute(f"CREATE OR REPLACE TABLE raw.{name} AS SELECT * FROM _raw_df")
        con.unregister("_raw_df")


def load_raw_parquet(con: Any, raw_dir: str | Path) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    for name in RAW_TABLES:
        path = (Path(raw_dir) / f"{name}.parquet").as_posix().replace("'", "''")
        con.execute(f"CREATE OR REPLACE TABLE raw.{name} AS SELECT * FROM read_parquet('{path}')")


def build_warehouse(
    con: Any,
    cfg: SimConfig,
    models_dir: str | Path | None = None,
    *,
    test: bool = True,
    fail_on_test_error: bool = True,
) -> RunResult:
    """Run all models (raw schema must already be loaded) and their data tests."""
    mdir = Path(models_dir) if models_dir else default_models_dir()
    variables = warehouse_vars(cfg)
    result = run_models(con, mdir, variables)
    if test:
        result.tests = run_tests(con, mdir, variables)
        failed = result.failed_tests
        if failed and fail_on_test_error:
            names = ", ".join(f"{t.test.name} ({t.failures} rows)" for t in failed)
            raise DataTestError(f"{len(failed)} data test(s) failed: {names}")
    return result
