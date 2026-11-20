"""Criteo Attribution Modeling for Bidding dataset -> user-level journeys.

Source: https://huggingface.co/datasets/criteo/criteo-attribution-dataset (CC BY-NC-SA 4.0;
Diemert et al. 2017, arXiv:1707.06409). One row per display impression with ``timestamp``
(seconds from the start of the 30-day window), ``uid``, ``campaign``, ``conversion``,
``conversion_timestamp``, ``conversion_id``, ``attribution`` (Criteo's own last-click label,
set on the conversion's impressions), ``click`` and more.

Journey definition (all done in DuckDB, so the 16.5M rows never become Python objects):

* A **conversion event** is a distinct ``(uid, conversion_id)`` with its conversion time.
* Each impression is assigned, per user, to the **first conversion event at or after it**
  (an ASOF join). Impressions after a user's last conversion form one non-converting journey.
  So a user's timeline is cut at every conversion, and a journey can mix campaigns.
* A **channel** is a campaign: the ``top_k`` campaigns by impression volume keep their own
  label (``c01`` = largest) and every other campaign is bucketed as ``other``. This keeps
  the channel set small enough for exact Shapley values and a dense Markov chain.
  ``channel_mode="touch_type"`` instead labels every impression ``click`` or ``view``.

Memory: converted journeys (one per conversion) are materialised with their touch times
(time decay needs them). Non-converting journeys are deduplicated to unique channel paths
plus a ``(path_id, user)`` index, and :meth:`CriteoJourneys.expand` repeats references to
the same immutable :class:`Journey`, so millions of journeys cost a few bytes each. Those
shared non-converting journeys carry zero ``days_before``: every model that reads them
(Markov, Shapley) ignores timing, and the heuristics only read converted journeys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import numpy.typing as npt

from gal.attribution.base import Journey

OTHER = "other"
SECONDS_PER_DAY = 86_400.0
ChannelMode = Literal["campaign", "touch_type"]
IntArray = npt.NDArray[np.int64]

CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/"
    "criteo_attribution_dataset.tsv.gz"
)

_COLUMNS = """
    timestamp::BIGINT AS ts, uid::BIGINT AS uid, campaign::BIGINT AS campaign,
    conversion::TINYINT AS conversion, conversion_timestamp::BIGINT AS conv_ts,
    conversion_id::BIGINT AS conv_id, attribution::TINYINT AS attribution,
    click::TINYINT AS click
"""


def tsv_to_parquet(tsv_gz: str | Path, parquet: str | Path) -> None:
    """Keep the columns the journey builder needs, as a compact Parquet file."""
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT {_COLUMNS} FROM read_csv(?, delim='\t', header=true)) "
        f"TO '{Path(parquet).as_posix()}' (FORMAT parquet)",
        [str(tsv_gz)],
    )
    con.close()


def register_source(con: duckdb.DuckDBPyConnection, parquet: str | Path) -> None:
    """Expose a Parquet file written by :func:`tsv_to_parquet` as the view ``criteo_src``."""
    con.execute(f"CREATE OR REPLACE VIEW criteo_src AS SELECT * FROM '{Path(parquet).as_posix()}'")


def register_frame(con: duckdb.DuckDBPyConnection, frame: Any) -> None:
    """Expose an in-memory DataFrame with the raw Criteo column names as ``criteo_src``."""
    con.register("criteo_raw_frame", frame)
    con.execute(f"CREATE OR REPLACE VIEW criteo_src AS SELECT {_COLUMNS} FROM criteo_raw_frame")


@dataclass
class CriteoJourneys:
    """Journeys for one channel definition, plus the user index needed to bootstrap."""

    channels: tuple[str, ...]
    converted: list[Journey]
    converted_user: IntArray
    owner_channel: list[str]
    """Channel of the campaign the conversion_id belongs to (its last labelled impression)."""
    criteo_channel: list[str | None]
    """Channel of Criteo's last-click attributed campaign; None if not attributed to Criteo."""
    nonconv_paths: list[Journey]
    nonconv_path_id: IntArray
    nonconv_user: IntArray
    n_users: int
    stats: dict[str, float] = field(default_factory=dict)

    def path_counts(self) -> IntArray:
        return np.bincount(self.nonconv_path_id, minlength=len(self.nonconv_paths)).astype(np.int64)

    def expand(
        self, conv_counts: IntArray | None = None, path_counts: IntArray | None = None
    ) -> list[Journey]:
        """All journeys, each repeated ``count`` times (references, not copies)."""
        cc = np.ones(len(self.converted), dtype=np.int64) if conv_counts is None else conv_counts
        pc = self.path_counts() if path_counts is None else path_counts
        out: list[Journey] = []
        for j, c in zip(self.converted, cc.tolist(), strict=True):
            if c:
                out.extend([j] * c)
        for j, c in zip(self.nonconv_paths, pc.tolist(), strict=True):
            if c:
                out.extend([j] * c)
        return out

    def bootstrap_counts(self, rng: np.random.Generator) -> tuple[IntArray, IntArray]:
        """Poisson(1) weights per *user*, applied to all of that user's journeys."""
        w = rng.poisson(1.0, self.n_users).astype(np.int64)
        conv = w[self.converted_user]
        paths = np.bincount(
            self.nonconv_path_id, weights=w[self.nonconv_user], minlength=len(self.nonconv_paths)
        )
        return conv, np.rint(paths).astype(np.int64)


def _channel_sql(mode: ChannelMode) -> str:
    if mode == "campaign":
        return f"coalesce(k.label, '{OTHER}')"
    if mode == "touch_type":
        return "CASE WHEN s.click = 1 THEN 'click' ELSE 'view' END"
    raise ValueError(f"unknown channel_mode {mode!r}")


def build_tables(
    con: duckdb.DuckDBPyConnection, top_k: int = 10, channel_mode: ChannelMode = "campaign"
) -> None:
    """Create ``topk``, ``users``, ``events``, ``journeys`` and ``paths`` from ``criteo_src``."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE topk AS
        SELECT campaign, impressions, rnk, printf('c%02d', rnk) AS label FROM (
            SELECT campaign, count(*) AS impressions,
                   row_number() OVER (ORDER BY count(*) DESC, campaign) AS rnk
            FROM criteo_src GROUP BY campaign)
        WHERE rnk <= ?
        """,
        [top_k],
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE users AS
        SELECT uid, row_number() OVER (ORDER BY uid) - 1 AS idx
        FROM (SELECT DISTINCT uid FROM criteo_src)
        """
    )
    # One row per conversion event. Criteo's label: the conversion is attributed to Criteo
    # (attribution = 1); its campaign is the last clicked impression carrying the label.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT uid, conv_id, min(conv_ts) AS conv_ts,
               arg_max(campaign, ts) AS owner_campaign,
               CASE WHEN max(attribution) = 1 THEN coalesce(
                   arg_max(campaign, ts) FILTER (WHERE click = 1 AND attribution = 1),
                   arg_max(campaign, ts) FILTER (WHERE click = 1),
                   arg_max(campaign, ts)) END AS criteo_campaign
        FROM criteo_src WHERE conv_id <> -1
        GROUP BY uid, conv_id
        """
    )
    end_ts = con.execute("SELECT max(ts) FROM criteo_src").fetchone()
    end = 0 if end_ts is None or end_ts[0] is None else int(end_ts[0])
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE journeys AS
        WITH touches AS (
            SELECT s.uid, s.ts, s.campaign, s.click, {_channel_sql(channel_mode)} AS channel
            FROM criteo_src s LEFT JOIN topk k USING (campaign)
        ), assigned AS (
            SELECT t.*, e.conv_id AS j_conv, e.conv_ts AS j_conv_ts
            FROM touches t ASOF LEFT JOIN events e ON t.uid = e.uid AND t.ts <= e.conv_ts
        )
        SELECT uid, coalesce(j_conv, -1) AS conv_id,
               list(channel ORDER BY ts, campaign, click) AS channels,
               list((coalesce(j_conv_ts, {end}) - ts) / {SECONDS_PER_DAY}
                    ORDER BY ts, campaign, click) AS days_before
        FROM assigned GROUP BY uid, coalesce(j_conv, -1)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE paths AS
        SELECT channels, row_number() OVER (ORDER BY channels) - 1 AS path_id
        FROM (SELECT DISTINCT channels FROM journeys WHERE conv_id = -1)
        """
    )


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    row = con.execute(sql).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def load_journeys(
    con: duckdb.DuckDBPyConnection, top_k: int = 10, channel_mode: ChannelMode = "campaign"
) -> CriteoJourneys:
    """Build the tables (see :func:`build_tables`) and pull compact journeys into Python."""
    build_tables(con, top_k, channel_mode)
    labels = [r[0] for r in con.execute("SELECT label FROM topk ORDER BY rnk").fetchall()]
    channels: tuple[str, ...] = (
        (*labels, OTHER) if channel_mode == "campaign" else ("click", "view")
    )

    rows = con.execute(
        f"""
        SELECT u.idx, e.uid, e.conv_id,
               coalesce(j.channels, []::VARCHAR[]) AS channels,
               coalesce(j.days_before, []::DOUBLE[]) AS days_before,
               coalesce(ko.label, '{OTHER}') AS owner,
               CASE WHEN e.criteo_campaign IS NULL THEN NULL
                    ELSE coalesce(kc.label, '{OTHER}') END AS criteo
        FROM events e
        JOIN users u USING (uid)
        LEFT JOIN journeys j ON j.uid = e.uid AND j.conv_id = e.conv_id
        LEFT JOIN topk ko ON ko.campaign = e.owner_campaign
        LEFT JOIN topk kc ON kc.campaign = e.criteo_campaign
        ORDER BY e.uid, e.conv_ts, e.conv_id
        """
    ).fetchall()
    converted = [
        Journey(f"{uid}:{cid}", tuple(ch), tuple(float(d) for d in db), True)
        for _, uid, cid, ch, db, _, _ in rows
    ]
    converted_user = np.array([r[0] for r in rows], dtype=np.int64)
    owner = [str(r[5]) for r in rows]
    criteo = [None if r[6] is None else str(r[6]) for r in rows]
    del rows

    path_rows = con.execute("SELECT path_id, channels FROM paths ORDER BY path_id").fetchall()
    nonconv_paths = [
        Journey(f"path{pid}", tuple(ch), (0.0,) * len(ch), False) for pid, ch in path_rows
    ]
    idx = con.execute(
        """
        SELECT p.path_id, u.idx FROM journeys j
        JOIN paths p USING (channels) JOIN users u USING (uid)
        WHERE j.conv_id = -1
        """
    ).fetchnumpy()
    n_users = int(_scalar(con, "SELECT count(*) FROM users"))

    multi = sum(1 for j in converted if len(set(j.channels)) > 1)
    stats = {
        "impressions": _scalar(con, "SELECT count(*) FROM criteo_src"),
        "users": float(n_users),
        "campaigns": _scalar(con, "SELECT count(DISTINCT campaign) FROM criteo_src"),
        "clicks": _scalar(con, "SELECT sum(click) FROM criteo_src"),
        "top_k": float(top_k),
        "top_k_impression_share": _scalar(
            con,
            "SELECT (SELECT sum(impressions) FROM topk) / (SELECT count(*) FROM criteo_src)",
        ),
        "conversions": float(len(converted)),
        "conversions_criteo_attributed": float(sum(c is not None for c in criteo)),
        "converted_journeys_empty": float(sum(j.is_empty for j in converted)),
        "converted_journeys_multi_channel": float(multi),
        "conversion_ids_multi_campaign": _scalar(
            con,
            """SELECT count(*) FROM (SELECT conv_id FROM criteo_src WHERE conv_id <> -1
               GROUP BY conv_id HAVING count(DISTINCT campaign) > 1)""",
        ),
        "conversion_ids_with_click": _scalar(
            con,
            """SELECT count(*) FROM (SELECT conv_id FROM criteo_src WHERE conv_id <> -1
               GROUP BY conv_id HAVING max(click) = 1)""",
        ),
        "conversion_ids": _scalar(
            con, "SELECT count(DISTINCT conv_id) FROM criteo_src WHERE conv_id <> -1"
        ),
        "mean_touches_converted": float(np.mean([len(j.channels) for j in converted]))
        if converted
        else 0.0,
        "nonconverting_journeys": float(len(idx["path_id"])),
        "unique_nonconverting_paths": float(len(nonconv_paths)),
    }
    return CriteoJourneys(
        channels=channels,
        converted=converted,
        converted_user=converted_user,
        owner_channel=owner,
        criteo_channel=criteo,
        nonconv_paths=nonconv_paths,
        nonconv_path_id=np.asarray(idx["path_id"], dtype=np.int64),
        nonconv_user=np.asarray(idx["idx"], dtype=np.int64),
        n_users=n_users,
        stats=stats,
    )


def label_counts(labels: Sequence[str | None], channels: Sequence[str]) -> dict[str, float]:
    """Count non-null labels per channel (e.g. Criteo's attributed conversions)."""
    out = dict.fromkeys(channels, 0.0)
    for lab in labels:
        if lab is not None:
            out[lab] = out.get(lab, 0.0) + 1.0
    return out


def weighted_label_counts(
    labels: Sequence[str | None], weights: IntArray, channels: Sequence[str]
) -> dict[str, float]:
    out = dict.fromkeys(channels, 0.0)
    for lab, w in zip(labels, weights.tolist(), strict=True):
        if lab is not None and w:
            out[lab] = out.get(lab, 0.0) + float(w)
    return out
