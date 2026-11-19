from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from gal.config import (
    SUPPORTED_CHANNELS,
    ConfigError,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert set(cfg.channel_names) == set(SUPPORTED_CHANNELS)
    assert all(len(c.lift) == 4 for c in cfg.channels)


def test_positive_control_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "positive_control.yaml")
    assert cfg.confounding == 0.0
    assert cfg.noise.missing_utm_rate == 0.0


def test_round_trip(tmp_path: Path, default_dict: dict[str, Any]) -> None:
    cfg = config_from_dict(default_dict)
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
    assert config_from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("channels", "paid", "lift"), [0.1, 0.2, 1.5, 0.0], "lift"),
        (("channels", "paid", "lift"), [0.1, 0.2], "4 entries"),
        (("noise", "identify_rate"), 1.2, "identify_rate"),
        (("n_accounts",), 3, "n_accounts"),
        (("data_end_day",), 10, "data_end_day"),
        (("confounding",), -1.0, "confounding"),
    ],
)
def test_invalid_values_rejected(
    default_dict: dict[str, Any], path: tuple[str, ...], value: Any, message: str
) -> None:
    d = copy.deepcopy(default_dict)
    node = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(d)


def test_unknown_channel_rejected(default_dict: dict[str, Any]) -> None:
    d = copy.deepcopy(default_dict)
    d["channels"]["tiktok"] = d["channels"]["paid"]
    with pytest.raises(ConfigError, match="unsupported"):
        config_from_dict(d)


def test_missing_key_is_config_error(default_dict: dict[str, Any]) -> None:
    d = copy.deepcopy(default_dict)
    del d["noise"]
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict(d)


def test_with_overrides_validates(default_dict: dict[str, Any]) -> None:
    cfg = config_from_dict(default_dict)
    assert cfg.with_overrides(seed=3).seed == 3
    with pytest.raises(ConfigError):
        cfg.with_overrides(confounding=-0.5)
