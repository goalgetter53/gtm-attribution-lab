from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from gal.config import SimConfig, config_from_dict
from gal.pipeline import PipelineRun, run_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_dict() -> dict[str, Any]:
    with (ROOT / "configs" / "default.yaml").open() as fh:
        d: dict[str, Any] = yaml.safe_load(fh)
    return d


@pytest.fixture(scope="session")
def small_cfg(default_dict: dict[str, Any]) -> SimConfig:
    d = dict(default_dict)
    d["n_accounts"] = 400
    d["gt_replicates"] = 6
    d["seed"] = 11
    return config_from_dict(d)


@pytest.fixture(scope="session")
def small_run(small_cfg: SimConfig) -> PipelineRun:
    return run_pipeline(small_cfg)
