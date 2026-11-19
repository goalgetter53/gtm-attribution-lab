"""End-to-end orchestration: generate -> warehouse -> journeys -> attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from gal.attribution import Credits, Journey, run_all
from gal.config import SimConfig
from gal.generator import GeneratedData, generate
from gal.warehouse.build import build_warehouse, load_raw_frames
from gal.warehouse.runner import RunResult

Level = Literal["account", "contact"]
LEVELS: tuple[Level, ...] = ("account", "contact")

_JOURNEY_SQL = """
select
    o.account_id,
    o.converted,
    (epoch(o.conversion_at) - epoch(o.mql_at)) / 86400.0 as mql_days_before,
    (epoch(o.conversion_at) - epoch(o.opportunity_at)) / 86400.0 as opp_days_before,
    (epoch(o.conversion_at) - epoch(o.lead_at)) / 86400.0 as lead_days_before,
    p.channel,
    p.days_before_conversion,
    p.is_focus_contact,
    p.touch_seq
from marts.fct_account_outcomes as o
left join marts.fct_attribution_paths as p on p.account_id = o.account_id
order by o.account_id, p.touch_seq
"""


def _opt(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def load_journeys(con: Any, level: Level = "account") -> list[Journey]:
    """Build one journey per account from the attribution-path mart.

    ``account`` level uses every stitched touch of the buying committee; ``contact`` level
    keeps only the focus contact's touches (what a lead-based/contact-based tool sees).
    """
    df = con.execute(_JOURNEY_SQL).df()
    journeys: list[Journey] = []
    for account_id, g in df.groupby("account_id", sort=True):
        first = g.iloc[0]
        touches = g[g["channel"].notna()]
        if level == "contact":
            touches = touches[touches["is_focus_contact"].fillna(False).astype(bool)]
        journeys.append(
            Journey(
                journey_id=str(account_id),
                channels=tuple(str(c) for c in touches["channel"]),
                days_before=tuple(float(d) for d in touches["days_before_conversion"]),
                converted=bool(first["converted"]),
                mql_days_before=_opt(first["mql_days_before"]),
                opp_days_before=_opt(first["opp_days_before"]),
                lead_days_before=_opt(first["lead_days_before"]),
            )
        )
    return journeys


@dataclass
class PipelineRun:
    cfg: SimConfig
    data: GeneratedData
    con: Any
    build: RunResult
    journeys: dict[str, list[Journey]] = field(default_factory=dict)
    credits: dict[str, dict[str, Credits]] = field(default_factory=dict)


def run_pipeline(
    cfg: SimConfig,
    models_dir: str | Path | None = None,
    *,
    attribute: bool = True,
    database: str = ":memory:",
) -> PipelineRun:
    data = generate(cfg)
    con = duckdb.connect(database)
    load_raw_frames(con, data.tables)
    result = build_warehouse(con, cfg, models_dir)
    run = PipelineRun(cfg=cfg, data=data, con=con, build=result)
    if attribute:
        for level in LEVELS:
            js = load_journeys(con, level)
            run.journeys[level] = js
            run.credits[level] = run_all(js, cfg.attribution, seed=cfg.seed)
    return run
