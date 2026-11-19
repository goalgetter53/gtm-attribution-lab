"""Typed configuration for the synthetic B2B world and the analytics pipeline.

Everything that shapes the simulation (channel effects, confounding, noise) and the
analysis (attribution window, Markov order, Shapley settings) lives in one YAML file so
that a run is fully reproducible from ``(config, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

STAGES: tuple[str, ...] = ("lead", "mql", "sql", "opportunity", "won")
"""Funnel stages in order. Index 0 is the entry stage; index 4 is the conversion."""

N_TRANSITIONS = len(STAGES) - 1

SUPPORTED_CHANNELS: tuple[str, ...] = (
    "outbound_email",
    "linkedin",
    "paid",
    "webinar",
    "organic",
    "sdr_call",
    "events",
)
"""Channels the generator knows how to render into raw source tables."""


@dataclass(frozen=True)
class ChannelConfig:
    """One marketing/sales channel and its *true* causal effect.

    ``lift[s]`` is the probability that a single touch of this channel moves an account
    from stage ``s`` to ``s + 1`` (lead->MQL, MQL->SQL, SQL->opp, opp->won). This is the
    known ground truth the attribution models are scored against.
    """

    name: str
    rate: float
    timing: tuple[float, float]
    intent_loading: float
    abm_loading: float
    lift: tuple[float, float, float, float]


@dataclass(frozen=True)
class BaselineConfig:
    """Unobserved, non-marketing progression (word of mouth, budget cycles, ...)."""

    rate: float
    lift: tuple[float, float, float, float]
    intent_effect: float
    direct_visit_prob: float


@dataclass(frozen=True)
class NoiseConfig:
    missing_utm_rate: float
    identify_rate: float
    personal_email_rate: float
    stage_lag_mean_days: float
    max_pageviews: int


@dataclass(frozen=True)
class CommitteeConfig:
    min_size: int
    mean_extra: float
    max_size: int
    champion_weight: float


@dataclass(frozen=True)
class AttributionConfig:
    window_days: int
    half_life_days: float
    markov_order: int
    shapley_exact_max_channels: int
    shapley_mc_permutations: int
    session_timeout_minutes: int


@dataclass(frozen=True)
class SimConfig:
    seed: int
    n_accounts: int
    cohort_span_days: int
    journey_days: int
    data_end_day: int
    confounding: float
    gt_replicates: int
    channels: tuple[ChannelConfig, ...]
    baseline: BaselineConfig
    noise: NoiseConfig
    committee: CommitteeConfig
    attribution: AttributionConfig

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    def with_overrides(self, **kwargs: Any) -> SimConfig:
        """Return a copy with top-level fields replaced (e.g. ``seed``, ``confounding``)."""
        new = replace(self, **kwargs)
        _validate(new)
        return new


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically invalid."""


def _prob(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {value}")


def _validate(cfg: SimConfig) -> None:
    if cfg.n_accounts < 10:
        raise ConfigError("n_accounts must be >= 10")
    if cfg.journey_days <= 0 or cfg.cohort_span_days <= 0:
        raise ConfigError("journey_days and cohort_span_days must be positive")
    if cfg.data_end_day <= cfg.cohort_span_days:
        raise ConfigError("data_end_day must be after the last lead cohort")
    if cfg.confounding < 0:
        raise ConfigError("confounding must be >= 0")
    if cfg.gt_replicates < 1:
        raise ConfigError("gt_replicates must be >= 1")
    names = [c.name for c in cfg.channels]
    if len(names) != len(set(names)):
        raise ConfigError("channel names must be unique")
    if not names:
        raise ConfigError("at least one channel is required")
    unknown = sorted(set(names) - set(SUPPORTED_CHANNELS))
    if unknown:
        raise ConfigError(f"unsupported channels {unknown}; choose from {SUPPORTED_CHANNELS}")
    for ch in cfg.channels:
        if ch.rate < 0:
            raise ConfigError(f"channel {ch.name}: rate must be >= 0")
        if min(ch.timing) <= 0:
            raise ConfigError(f"channel {ch.name}: timing Beta parameters must be > 0")
        for i, p in enumerate(ch.lift):
            _prob(f"channel {ch.name} lift[{i}]", p)
    for i, p in enumerate(cfg.baseline.lift):
        _prob(f"baseline lift[{i}]", p)
    _prob("baseline.direct_visit_prob", cfg.baseline.direct_visit_prob)
    _prob("noise.missing_utm_rate", cfg.noise.missing_utm_rate)
    _prob("noise.identify_rate", cfg.noise.identify_rate)
    _prob("noise.personal_email_rate", cfg.noise.personal_email_rate)
    _prob("committee.champion_weight", cfg.committee.champion_weight)
    if not 1 <= cfg.committee.min_size <= cfg.committee.max_size:
        raise ConfigError("committee sizes must satisfy 1 <= min_size <= max_size")
    if cfg.attribution.markov_order < 1:
        raise ConfigError("markov_order must be >= 1")
    if cfg.attribution.window_days <= 0:
        raise ConfigError("attribution window_days must be positive")


def _four(name: str, values: Any) -> tuple[float, float, float, float]:
    vals = [float(v) for v in values]
    if len(vals) != N_TRANSITIONS:
        raise ConfigError(f"{name} must have {N_TRANSITIONS} entries (one per transition)")
    return (vals[0], vals[1], vals[2], vals[3])


def config_from_dict(raw: dict[str, Any]) -> SimConfig:
    """Build and validate a :class:`SimConfig` from a parsed YAML mapping."""
    try:
        channels = tuple(
            ChannelConfig(
                name=str(name),
                rate=float(spec["rate"]),
                timing=(float(spec["timing"][0]), float(spec["timing"][1])),
                intent_loading=float(spec.get("intent_loading", 0.0)),
                abm_loading=float(spec.get("abm_loading", 0.0)),
                lift=_four(f"channel {name} lift", spec["lift"]),
            )
            for name, spec in raw["channels"].items()
        )
        b = raw["baseline"]
        n = raw["noise"]
        c = raw["committee"]
        a = raw["attribution"]
        cfg = SimConfig(
            seed=int(raw["seed"]),
            n_accounts=int(raw["n_accounts"]),
            cohort_span_days=int(raw["cohort_span_days"]),
            journey_days=int(raw["journey_days"]),
            data_end_day=int(raw["data_end_day"]),
            confounding=float(raw["confounding"]),
            gt_replicates=int(raw.get("gt_replicates", 32)),
            channels=channels,
            baseline=BaselineConfig(
                rate=float(b["rate"]),
                lift=_four("baseline lift", b["lift"]),
                intent_effect=float(b["intent_effect"]),
                direct_visit_prob=float(b.get("direct_visit_prob", 0.3)),
            ),
            noise=NoiseConfig(
                missing_utm_rate=float(n["missing_utm_rate"]),
                identify_rate=float(n["identify_rate"]),
                personal_email_rate=float(n["personal_email_rate"]),
                stage_lag_mean_days=float(n["stage_lag_mean_days"]),
                max_pageviews=int(n.get("max_pageviews", 4)),
            ),
            committee=CommitteeConfig(
                min_size=int(c["min_size"]),
                mean_extra=float(c["mean_extra"]),
                max_size=int(c["max_size"]),
                champion_weight=float(c.get("champion_weight", 0.5)),
            ),
            attribution=AttributionConfig(
                window_days=int(a["window_days"]),
                half_life_days=float(a["half_life_days"]),
                markov_order=int(a.get("markov_order", 1)),
                shapley_exact_max_channels=int(a.get("shapley_exact_max_channels", 10)),
                shapley_mc_permutations=int(a.get("shapley_mc_permutations", 2000)),
                session_timeout_minutes=int(a.get("session_timeout_minutes", 30)),
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"missing config key: {exc}") from exc
    _validate(cfg)
    return cfg


def load_config(path: str | Path) -> SimConfig:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} does not contain a mapping")
    return config_from_dict(raw)


def config_to_dict(cfg: SimConfig) -> dict[str, Any]:
    """Inverse of :func:`config_from_dict` (round-trippable)."""
    return {
        "seed": cfg.seed,
        "n_accounts": cfg.n_accounts,
        "cohort_span_days": cfg.cohort_span_days,
        "journey_days": cfg.journey_days,
        "data_end_day": cfg.data_end_day,
        "confounding": cfg.confounding,
        "gt_replicates": cfg.gt_replicates,
        "channels": {
            ch.name: {
                "rate": ch.rate,
                "timing": list(ch.timing),
                "intent_loading": ch.intent_loading,
                "abm_loading": ch.abm_loading,
                "lift": list(ch.lift),
            }
            for ch in cfg.channels
        },
        "baseline": {
            "rate": cfg.baseline.rate,
            "lift": list(cfg.baseline.lift),
            "intent_effect": cfg.baseline.intent_effect,
            "direct_visit_prob": cfg.baseline.direct_visit_prob,
        },
        "noise": {
            "missing_utm_rate": cfg.noise.missing_utm_rate,
            "identify_rate": cfg.noise.identify_rate,
            "personal_email_rate": cfg.noise.personal_email_rate,
            "stage_lag_mean_days": cfg.noise.stage_lag_mean_days,
            "max_pageviews": cfg.noise.max_pageviews,
        },
        "committee": {
            "min_size": cfg.committee.min_size,
            "mean_extra": cfg.committee.mean_extra,
            "max_size": cfg.committee.max_size,
            "champion_weight": cfg.committee.champion_weight,
        },
        "attribution": {
            "window_days": cfg.attribution.window_days,
            "half_life_days": cfg.attribution.half_life_days,
            "markov_order": cfg.attribution.markov_order,
            "shapley_exact_max_channels": cfg.attribution.shapley_exact_max_channels,
            "shapley_mc_permutations": cfg.attribution.shapley_mc_permutations,
            "session_timeout_minutes": cfg.attribution.session_timeout_minutes,
        },
    }


def save_config(cfg: SimConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
