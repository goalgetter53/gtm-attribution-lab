"""Synthetic B2B journey generator with a KNOWN causal ground truth.

The data-generating process (DGP), per account ``a``:

1. Latent *intent* ``z_a ~ N(0, 1)`` (the confounder) and an *ABM-targeting* factor
   ``f_a ~ N(0, 1)`` that correlates LinkedIn, paid and events exposure.
2. For every channel ``c`` the number of touches is Poisson with a mean-preserving
   log-normal rate ``rate_c * exp(k*l_c*z_a + m_c*f_a - var/2)`` where ``k`` is the
   global ``confounding`` knob. Touch timing inside the journey follows ``Beta(a_c, b_c)``.
3. Unobserved *baseline* nudges (word of mouth, budget cycles) arrive as a Poisson
   process; their strength grows with intent.
4. Progression is a sequential noisy-OR state machine over
   lead -> MQL -> SQL -> opportunity -> won. Each event (touch or baseline nudge) at the
   account's current stage ``s`` advances it with probability ``lift[c, s]``.

Ground truth for channel ``c`` is the *counterfactual* expected number of wins lost if
every ``c`` touch were deleted, estimated with common random numbers over many
replicates. Attribution models only ever see the noisy raw tables rendered from the
realised world (missing UTMs, anonymous sessions, CRM stage lag, buying committees).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gal.config import N_TRANSITIONS, SimConfig

EPOCH = pd.Timestamp("2025-01-01")
BASELINE = -1
PAD = -2

WEB_CHANNELS = frozenset({"paid", "organic", "linkedin"})
ACTIVITY_TYPES: dict[str, str] = {
    "sdr_call": "call",
    "webinar": "webinar_attended",
    "events": "event_attended",
}
EMPLOYEE_BANDS: tuple[str, ...] = ("1-50", "51-200", "201-1000", "1000+")
BAND_ACV: dict[str, float] = {
    "1-50": 12_000,
    "51-200": 30_000,
    "201-1000": 60_000,
    "1000+": 120_000,
}
INDUSTRIES: tuple[str, ...] = (
    "fintech",
    "healthtech",
    "saas",
    "manufacturing",
    "retail",
    "logistics",
)
TITLES: tuple[str, ...] = (
    "VP Engineering",
    "Head of RevOps",
    "Director of IT",
    "CFO",
    "Security Lead",
    "Procurement Manager",
    "Product Manager",
    "Data Engineer",
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class World:
    """Latent simulation state (never exposed to attribution models)."""

    channels: tuple[str, ...]
    intent: FloatArray
    abm: FloatArray
    lead_day: FloatArray
    committee_size: IntArray
    event_channel: IntArray  # (n, K): channel index, BASELINE or PAD
    event_day: FloatArray  # (n, K): day offset from EPOCH, +inf for PAD
    event_contact: IntArray  # (n, K): contact index within the account
    base_lift: FloatArray  # (n, 4): baseline nudge lift per account and stage
    lift: FloatArray  # (C, 4): true per-touch lift per channel and stage

    @property
    def n_accounts(self) -> int:
        return int(self.intent.shape[0])


@dataclass(frozen=True)
class GroundTruth:
    """Counterfactual channel effects (expected wins removed if a channel is removed)."""

    channels: tuple[str, ...]
    expected_wins: float
    incremental_wins: dict[str, float]
    realized_wins: int
    replicates: int
    confounding: float
    seed: int

    @property
    def share(self) -> dict[str, float]:
        """Normalised share of incremental wins (negatives from MC noise clipped to 0)."""
        pos = {c: max(v, 0.0) for c, v in self.incremental_wins.items()}
        total = sum(pos.values())
        if total <= 0:
            return {c: 1.0 / len(pos) for c in pos}
        return {c: v / total for c, v in pos.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "expected_wins": self.expected_wins,
            "incremental_wins": self.incremental_wins,
            "share": self.share,
            "realized_wins": self.realized_wins,
            "replicates": self.replicates,
            "confounding": self.confounding,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroundTruth:
        return cls(
            channels=tuple(d["channels"]),
            expected_wins=float(d["expected_wins"]),
            incremental_wins={k: float(v) for k, v in d["incremental_wins"].items()},
            realized_wins=int(d["realized_wins"]),
            replicates=int(d["replicates"]),
            confounding=float(d["confounding"]),
            seed=int(d["seed"]),
        )


@dataclass
class GeneratedData:
    """Raw source tables plus the ground truth that produced them."""

    tables: dict[str, pd.DataFrame]
    ground_truth: GroundTruth
    world: World = field(repr=False)

    def write(self, out_dir: str | Path) -> None:
        """Persist raw tables as Parquet (via DuckDB) and the ground truth as JSON."""
        import duckdb

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        try:
            for name, df in self.tables.items():
                con.register("_df", df)
                target = (out / f"{name}.parquet").as_posix().replace("'", "''")
                con.execute(f"COPY _df TO '{target}' (FORMAT PARQUET)")
                con.unregister("_df")
        finally:
            con.close()
        (out / "ground_truth.json").write_text(
            json.dumps(self.ground_truth.to_dict(), indent=2), encoding="utf-8"
        )


RAW_TABLES: tuple[str, ...] = (
    "accounts",
    "contacts",
    "web_events",
    "identifies",
    "email_events",
    "activities",
    "opportunities",
    "stage_history",
)


# --------------------------------------------------------------------------- world


def build_world(cfg: SimConfig, rng: np.random.Generator) -> World:
    """Draw latent account traits and the full (touch + baseline) event schedule."""
    n = cfg.n_accounts
    C = len(cfg.channels)
    intent = rng.standard_normal(n)
    abm = rng.standard_normal(n)
    lead_day = rng.uniform(0.0, float(cfg.cohort_span_days), n)
    committee = np.minimum(
        cfg.committee.min_size + rng.poisson(cfg.committee.mean_extra, n),
        cfg.committee.max_size,
    ).astype(np.int64)

    acc_parts: list[IntArray] = []
    chan_parts: list[IntArray] = []
    day_parts: list[FloatArray] = []
    for ci, ch in enumerate(cfg.channels):
        k_load = cfg.confounding * ch.intent_loading
        log_rate = k_load * intent + ch.abm_loading * abm - 0.5 * (k_load**2 + ch.abm_loading**2)
        counts = rng.poisson(ch.rate * np.exp(log_rate))
        acc = np.repeat(np.arange(n, dtype=np.int64), counts)
        pos = rng.beta(ch.timing[0], ch.timing[1], acc.shape[0])
        acc_parts.append(acc)
        chan_parts.append(np.full(acc.shape[0], ci, dtype=np.int64))
        day_parts.append(lead_day[acc] + cfg.journey_days * pos)
    base_counts = rng.poisson(cfg.baseline.rate, n)
    acc_b = np.repeat(np.arange(n, dtype=np.int64), base_counts)
    acc_parts.append(acc_b)
    chan_parts.append(np.full(acc_b.shape[0], BASELINE, dtype=np.int64))
    day_parts.append(lead_day[acc_b] + rng.uniform(0.0, float(cfg.journey_days), acc_b.shape[0]))

    acc_all = np.concatenate(acc_parts)
    chan_all = np.concatenate(chan_parts)
    day_all = np.concatenate(day_parts)
    # Events after the data cutoff have not happened yet.
    keep = day_all <= cfg.data_end_day
    acc_all, chan_all, day_all = acc_all[keep], chan_all[keep], day_all[keep]

    # Contact who receives each event: the champion (index 0) or another committee member.
    size = committee[acc_all]
    champion = rng.random(acc_all.shape[0]) < cfg.committee.champion_weight
    other = 1 + np.floor(rng.random(acc_all.shape[0]) * np.maximum(size - 1, 1)).astype(np.int64)
    contact_all = np.where(champion | (size == 1), 0, np.minimum(other, size - 1))

    order = np.lexsort((day_all, acc_all))
    acc_all, chan_all, day_all, contact_all = (
        acc_all[order],
        chan_all[order],
        day_all[order],
        contact_all[order],
    )
    counts_per_acc = np.bincount(acc_all, minlength=n)
    K = max(int(np.max(counts_per_acc)) if n else 0, 1)
    starts = np.concatenate([[0], np.cumsum(counts_per_acc)[:-1]])
    pos_in_acc = np.arange(acc_all.shape[0]) - starts[acc_all]

    event_channel = np.full((n, K), PAD, dtype=np.int64)
    event_day = np.full((n, K), np.inf)
    event_contact = np.full((n, K), -1, dtype=np.int64)
    event_channel[acc_all, pos_in_acc] = chan_all
    event_day[acc_all, pos_in_acc] = day_all
    event_contact[acc_all, pos_in_acc] = contact_all

    base = np.asarray(cfg.baseline.lift, dtype=np.float64)
    strength = np.exp(cfg.baseline.intent_effect * intent)[:, None]
    base_lift = 1.0 - np.power(1.0 - base[None, :], strength)
    lift = np.asarray([ch.lift for ch in cfg.channels], dtype=np.float64).reshape(C, N_TRANSITIONS)

    return World(
        channels=cfg.channel_names,
        intent=intent,
        abm=abm,
        lead_day=lead_day,
        committee_size=committee,
        event_channel=event_channel,
        event_day=event_day,
        event_contact=event_contact,
        base_lift=base_lift,
        lift=lift,
    )


def simulate_progression(
    world: World, uniforms: FloatArray, removed: int | None = None
) -> tuple[IntArray, IntArray]:
    """Run the stage state machine for every account (vectorised over accounts).

    Args:
        world: latent world.
        uniforms: ``(n, K)`` U(0,1) draws; reuse the same draws across scenarios
            (common random numbers) to get low-variance counterfactual differences.
        removed: channel index whose touches are deleted (counterfactual), or None.

    Returns:
        ``(final_stage, trigger)`` where ``final_stage`` is in 0..4 and ``trigger[a, s]``
        is the event index that moved account ``a`` out of stage ``s`` (-1 if never).
    """
    n, K = world.event_channel.shape
    stage = np.zeros(n, dtype=np.int64)
    trigger = np.full((n, N_TRANSITIONS), -1, dtype=np.int64)
    rows = np.arange(n)
    for k in range(K):
        ch = world.event_channel[:, k]
        valid = ch != PAD
        if removed is not None:
            valid &= ch != removed
        if not valid.any():
            continue
        s = np.minimum(stage, N_TRANSITIONS - 1)
        p_touch = world.lift[np.clip(ch, 0, None), s]
        p = np.where(ch == BASELINE, world.base_lift[rows, s], p_touch)
        adv = valid & (stage < N_TRANSITIONS) & (uniforms[:, k] < p)
        idx = np.nonzero(adv)[0]
        trigger[idx, stage[idx]] = k
        stage = stage + adv.astype(np.int64)
    return stage, trigger


def compute_ground_truth(world: World, cfg: SimConfig, rng: np.random.Generator) -> GroundTruth:
    """Counterfactual removal effect per channel with common random numbers."""
    n, K = world.event_channel.shape
    C = len(world.channels)
    wins_all = 0.0
    delta = np.zeros(C)
    for _ in range(cfg.gt_replicates):
        u = rng.random((n, K))
        final, _ = simulate_progression(world, u)
        w_all = float((final == N_TRANSITIONS).sum())
        wins_all += w_all
        for c in range(C):
            final_c, _ = simulate_progression(world, u, removed=c)
            delta[c] += w_all - float((final_c == N_TRANSITIONS).sum())
    r = float(cfg.gt_replicates)
    return GroundTruth(
        channels=world.channels,
        expected_wins=wins_all / r,
        incremental_wins={ch: float(delta[i] / r) for i, ch in enumerate(world.channels)},
        realized_wins=0,
        replicates=cfg.gt_replicates,
        confounding=cfg.confounding,
        seed=cfg.seed,
    )


# --------------------------------------------------------------------------- raw tables


def _ts(days: FloatArray | pd.Series) -> pd.Series:
    return EPOCH + pd.to_timedelta(np.asarray(days, dtype=np.float64), unit="D")


def _stage_days(
    world: World, trigger: IntArray, rng: np.random.Generator, cfg: SimConfig
) -> FloatArray:
    """CRM-recorded day each stage was entered (NaN if never). Includes recording lag."""
    n = world.n_accounts
    days = np.full((n, N_TRANSITIONS + 1), np.nan)
    days[:, 0] = world.lead_day
    rows = np.arange(n)
    for s in range(N_TRANSITIONS):
        k = trigger[:, s]
        hit = k >= 0
        lag = rng.exponential(cfg.noise.stage_lag_mean_days, n)
        d = np.where(hit, world.event_day[rows, np.maximum(k, 0)] + lag, np.nan)
        d = np.minimum(d, float(cfg.data_end_day))
        prev = days[:, s]
        days[:, s + 1] = np.where(hit, np.fmax(d, prev), np.nan)
    return days


def render_tables(
    world: World,
    final: IntArray,
    trigger: IntArray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Render the realised world into messy raw source tables (CRM, web, MAP, sequencer)."""
    n = world.n_accounts
    noise = cfg.noise
    band_idx = rng.integers(0, len(EMPLOYEE_BANDS), n)
    industry_idx = rng.integers(0, len(INDUSTRIES), n)

    accounts = pd.DataFrame(
        {
            "account_id": [f"A{i:05d}" for i in range(n)],
            "domain": [f"acct{i:05d}.example.com" for i in range(n)],
            "industry": [INDUSTRIES[j] for j in industry_idx],
            "employee_band": [EMPLOYEE_BANDS[j] for j in band_idx],
            "created_at": _ts(world.lead_day),
        }
    )

    # ---- contacts (buying committee) and identity
    c_rows: list[dict[str, Any]] = []
    id_rows: list[dict[str, Any]] = []
    contact_ids: list[list[str]] = []
    contact_emails: list[list[str]] = []
    anon_ids: list[list[str]] = []
    for a in range(n):
        ids, emails, anons = [], [], []
        for j in range(int(world.committee_size[a])):
            cid = f"C{a:05d}_{j}"
            personal = rng.random() < noise.personal_email_rate
            email = (
                f"buyer.{a:05d}.{j}@gmail.com" if personal else f"person{j}@acct{a:05d}.example.com"
            )
            c_rows.append(
                {
                    "contact_id": cid,
                    # Personal emails are never linked; ~20% of business leads were never
                    # converted to CRM contacts and must be matched on email domain instead.
                    "account_id": None if personal or rng.random() < 0.2 else f"A{a:05d}",
                    "email": email.upper() if rng.random() < 0.05 else email,
                    "title": TITLES[int(rng.integers(0, len(TITLES)))],
                    "created_at_day": world.lead_day[a] + rng.uniform(0, 5),
                }
            )
            anon = f"anon_{a:05d}_{j}"
            if rng.random() < noise.identify_rate:
                id_rows.append(
                    {
                        "anonymous_id": anon,
                        "email": email,
                        "identify_day": world.lead_day[a] + rng.uniform(0, float(cfg.journey_days)),
                    }
                )
            ids.append(cid)
            emails.append(email)
            anons.append(anon)
        contact_ids.append(ids)
        contact_emails.append(emails)
        anon_ids.append(anons)
    contacts = pd.DataFrame(c_rows)
    contacts["created_at"] = _ts(contacts.pop("created_at_day"))
    identifies = pd.DataFrame(id_rows, columns=["anonymous_id", "email", "identify_day"])
    identifies["identified_at"] = _ts(identifies.pop("identify_day"))

    # ---- touches -> web / email / activity raw events
    web_rows: list[dict[str, Any]] = []
    email_rows: list[dict[str, Any]] = []
    act_rows: list[dict[str, Any]] = []
    rows_idx, cols_idx = np.nonzero(world.event_channel != PAD)
    for a, k in zip(rows_idx.tolist(), cols_idx.tolist(), strict=True):
        ch_i = int(world.event_channel[a, k])
        day = float(world.event_day[a, k])
        j = int(world.event_contact[a, k])
        anon = anon_ids[a][j]
        if ch_i == BASELINE:
            if rng.random() < cfg.baseline.direct_visit_prob:
                _web_session(web_rows, rng, anon, day, None, None, None, cfg)
            continue
        ch = world.channels[ch_i]
        if ch in WEB_CHANNELS:
            lost = rng.random() < noise.missing_utm_rate
            utm: tuple[str | None, str | None, str | None]
            if ch == "paid":
                utm = (None, None, None) if lost else ("google", "cpc", "https://www.google.com/")
            elif ch == "linkedin":
                utm = (
                    (None, None, None)
                    if lost
                    else ("linkedin", "paid_social", "https://www.linkedin.com/")
                )
            else:  # organic: identified by search referrer, never by UTM
                utm = (None, None, None) if lost else (None, None, "https://www.google.com/")
            _web_session(web_rows, rng, anon, day, utm[0], utm[1], utm[2], cfg)
        elif ch == "outbound_email":
            email = contact_emails[a][j]
            for _ in range(int(rng.integers(1, 4))):
                email_rows.append(
                    {"email": email, "event_type": "sent", "day": day - rng.uniform(0.5, 6)}
                )
            if rng.random() < 0.5:
                email_rows.append(
                    {"email": email, "event_type": "open", "day": day - rng.uniform(0, 0.4)}
                )
            email_rows.append(
                {
                    "email": email,
                    "event_type": "click" if rng.random() < 0.8 else "reply",
                    "day": day,
                }
            )
        else:
            act_rows.append(
                {
                    "contact_id": contact_ids[a][j],
                    "activity_type": ACTIVITY_TYPES[ch],
                    "outcome": "connected" if ch == "sdr_call" else "attended",
                    "day": day,
                }
            )
    # Non-touch noise: unanswered dials never reach the buyer.
    for _ in range(int(0.3 * len(act_rows))):
        a = int(rng.integers(0, n))
        act_rows.append(
            {
                "contact_id": contact_ids[a][0],
                "activity_type": "call",
                "outcome": "no_answer",
                "day": world.lead_day[a] + rng.uniform(0, float(cfg.journey_days)),
            }
        )

    web_events = pd.DataFrame(
        web_rows,
        columns=["anonymous_id", "day", "page", "utm_source", "utm_medium", "referrer"],
    )
    web_events = web_events[web_events["day"] <= cfg.data_end_day].reset_index(drop=True)
    web_events.insert(0, "event_id", [f"W{i:07d}" for i in range(len(web_events))])
    web_events["event_ts"] = _ts(web_events.pop("day"))

    email_events = pd.DataFrame(email_rows, columns=["email", "event_type", "day"])
    email_events.insert(0, "event_id", [f"E{i:07d}" for i in range(len(email_events))])
    email_events["event_ts"] = _ts(email_events.pop("day"))
    email_events["campaign"] = "outbound_q_sequence"

    activities = pd.DataFrame(act_rows, columns=["contact_id", "activity_type", "outcome", "day"])
    activities = activities[activities["day"] <= cfg.data_end_day].reset_index(drop=True)
    activities.insert(0, "activity_id", [f"T{i:07d}" for i in range(len(activities))])
    activities["activity_ts"] = _ts(activities.pop("day"))

    # ---- stage history and opportunities (CRM, with recording lag)
    stage_days = _stage_days(world, trigger, rng, cfg)
    stage_names = ("lead", "mql", "sql", "opportunity", "won")
    sh_rows: list[dict[str, Any]] = []
    opp_rows: list[dict[str, Any]] = []
    for a in range(n):
        for s, name in enumerate(stage_names):
            d = stage_days[a, s]
            if not np.isnan(d):
                sh_rows.append({"account_id": f"A{a:05d}", "stage": name, "day": d})
        opp_day = stage_days[a, 3]
        if np.isnan(opp_day):
            continue
        band = EMPLOYEE_BANDS[int(band_idx[a])]
        amount = round(BAND_ACV[band] * float(rng.lognormal(0.0, 0.35)), 2)
        size = int(world.committee_size[a])
        primary = 0 if rng.random() < 0.55 or size == 1 else int(rng.integers(1, size))
        won = int(final[a]) == N_TRANSITIONS
        if won:
            closed: float | None = float(stage_days[a, 4])
            stage_name = "closed_won"
        else:
            lost_day = opp_day + rng.uniform(30, 150)
            closed = lost_day if lost_day <= cfg.data_end_day else None
            stage_name = "closed_lost" if closed is not None else "open"
        opp_rows.append(
            {
                "opportunity_id": f"O{a:05d}",
                "account_id": f"A{a:05d}",
                "primary_contact_id": contact_ids[a][primary],
                "created_day": opp_day,
                "closed_day": closed,
                "stage_name": stage_name,
                "is_won": won,
                "amount": amount,
            }
        )
    stage_history = pd.DataFrame(sh_rows, columns=["account_id", "stage", "day"])
    stage_history["entered_at"] = _ts(stage_history.pop("day"))
    opportunities = pd.DataFrame(
        opp_rows,
        columns=[
            "opportunity_id",
            "account_id",
            "primary_contact_id",
            "created_day",
            "closed_day",
            "stage_name",
            "is_won",
            "amount",
        ],
    )
    opportunities["created_at"] = _ts(opportunities.pop("created_day"))
    closed_days = opportunities.pop("closed_day").astype(float)
    opportunities["closed_at"] = _ts(closed_days).where(closed_days.notna().to_numpy())

    return {
        "accounts": accounts,
        "contacts": contacts,
        "web_events": web_events,
        "identifies": identifies,
        "email_events": email_events,
        "activities": activities,
        "opportunities": opportunities,
        "stage_history": stage_history,
    }


def _web_session(
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    anon: str,
    day: float,
    utm_source: str | None,
    utm_medium: str | None,
    referrer: str | None,
    cfg: SimConfig,
) -> None:
    """Append a landing pageview (carrying attribution params) plus internal pageviews."""
    pages = ("/pricing", "/blog/attribution", "/product", "/case-studies", "/demo", "/docs")
    rows.append(
        {
            "anonymous_id": anon,
            "day": day,
            "page": pages[int(rng.integers(0, len(pages)))],
            "utm_source": utm_source,
            "utm_medium": utm_medium,
            "referrer": referrer,
        }
    )
    t = day
    for _ in range(int(rng.integers(0, cfg.noise.max_pageviews))):
        t += rng.uniform(0.5, 8.0) / (24 * 60)  # 0.5-8 minutes later
        rows.append(
            {
                "anonymous_id": anon,
                "day": t,
                "page": pages[int(rng.integers(0, len(pages)))],
                "utm_source": None,
                "utm_medium": None,
                "referrer": "https://www.gtmlab-vendor.com/",
            }
        )


def generate(cfg: SimConfig) -> GeneratedData:
    """Generate a full synthetic dataset for ``cfg`` (deterministic in ``cfg.seed``)."""
    root = np.random.SeedSequence(cfg.seed)
    world_ss, real_ss, render_ss, gt_ss = root.spawn(4)
    world = build_world(cfg, np.random.default_rng(world_ss))
    real_rng = np.random.default_rng(real_ss)
    u = real_rng.random(world.event_channel.shape)
    final, trigger = simulate_progression(world, u)
    tables = render_tables(world, final, trigger, cfg, np.random.default_rng(render_ss))
    gt = compute_ground_truth(world, cfg, np.random.default_rng(gt_ss))
    gt = GroundTruth(
        channels=gt.channels,
        expected_wins=gt.expected_wins,
        incremental_wins=gt.incremental_wins,
        realized_wins=int((final == N_TRANSITIONS).sum()),
        replicates=gt.replicates,
        confounding=gt.confounding,
        seed=gt.seed,
    )
    return GeneratedData(tables=tables, ground_truth=gt, world=world)


def load_ground_truth(path: str | Path) -> GroundTruth:
    return GroundTruth.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
