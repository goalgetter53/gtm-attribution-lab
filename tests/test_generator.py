from __future__ import annotations

import numpy as np
import pandas as pd

from gal.config import SimConfig
from gal.generator import (
    BASELINE,
    PAD,
    RAW_TABLES,
    GroundTruth,
    build_world,
    generate,
    simulate_progression,
)


def test_generation_is_deterministic(small_cfg: SimConfig) -> None:
    a = generate(small_cfg)
    b = generate(small_cfg)
    for name in RAW_TABLES:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])
    assert a.ground_truth == b.ground_truth


def test_different_seed_changes_data(small_cfg: SimConfig) -> None:
    a = generate(small_cfg)
    b = generate(small_cfg.with_overrides(seed=small_cfg.seed + 1))
    assert not a.tables["web_events"].equals(b.tables["web_events"])


def test_all_raw_tables_present(small_cfg: SimConfig) -> None:
    data = generate(small_cfg)
    assert set(data.tables) == set(RAW_TABLES)
    assert len(data.tables["accounts"]) == small_cfg.n_accounts


def test_events_sorted_and_padded(small_cfg: SimConfig) -> None:
    w = build_world(small_cfg, np.random.default_rng(0))
    real = w.event_channel != PAD
    # padding only at the end of each row, and event days non-decreasing
    assert np.all(np.diff(real.astype(int), axis=1) <= 0)
    days = np.where(real, w.event_day, np.nan)
    diffs = np.diff(days, axis=1)
    assert np.all((diffs >= 0) | np.isnan(diffs))
    assert np.nanmax(days) <= small_cfg.data_end_day


def test_committee_sizes_within_bounds(small_cfg: SimConfig) -> None:
    w = build_world(small_cfg, np.random.default_rng(1))
    assert w.committee_size.min() >= small_cfg.committee.min_size
    assert w.committee_size.max() <= small_cfg.committee.max_size
    real = w.event_channel != PAD
    assert np.all(w.event_contact[real] < w.committee_size[np.nonzero(real)[0]])


def test_confounding_correlates_exposure_with_intent(small_cfg: SimConfig) -> None:
    cfg = small_cfg.with_overrides(n_accounts=3000, confounding=1.5)
    w = build_world(cfg, np.random.default_rng(2))
    organic = cfg.channel_names.index("organic")
    counts = (w.event_channel == organic).sum(axis=1)
    assert np.corrcoef(w.intent, counts)[0, 1] > 0.3
    w0 = build_world(cfg.with_overrides(confounding=0.0), np.random.default_rng(2))
    counts0 = (w0.event_channel == organic).sum(axis=1)
    assert abs(np.corrcoef(w0.intent, counts0)[0, 1]) < 0.1


def test_progression_is_monotone_and_bounded(small_cfg: SimConfig) -> None:
    w = build_world(small_cfg, np.random.default_rng(3))
    u = np.random.default_rng(4).random(w.event_channel.shape)
    final, trigger = simulate_progression(w, u)
    assert final.min() >= 0 and final.max() <= 4
    for a in range(w.n_accounts):
        trig = trigger[a, : final[a]]
        assert np.all(trig >= 0) and np.all(np.diff(trig) > 0)
        assert np.all(trigger[a, final[a] :] == -1)


def test_zero_lift_world_only_converts_via_baseline(small_cfg: SimConfig) -> None:
    w = build_world(small_cfg, np.random.default_rng(5))
    w_nolift = type(w)(**{**w.__dict__, "lift": np.zeros_like(w.lift)})
    u = np.random.default_rng(6).random(w.event_channel.shape)
    _, trigger = simulate_progression(w_nolift, u)
    rows, stages = np.nonzero(trigger >= 0)
    assert np.all(w.event_channel[rows, trigger[rows, stages]] == BASELINE)


def test_removing_a_channel_never_adds_its_triggers(small_cfg: SimConfig) -> None:
    w = build_world(small_cfg, np.random.default_rng(7))
    u = np.random.default_rng(8).random(w.event_channel.shape)
    _, trigger = simulate_progression(w, u, removed=0)
    rows, stages = np.nonzero(trigger >= 0)
    assert not np.any(w.event_channel[rows, trigger[rows, stages]] == 0)


def test_ground_truth_shares_and_serialisation(small_cfg: SimConfig) -> None:
    gt = generate(small_cfg).ground_truth
    assert abs(sum(gt.share.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in gt.share.values())
    assert GroundTruth.from_dict(gt.to_dict()) == gt


def test_zero_effect_channel_has_near_zero_truth(small_cfg: SimConfig) -> None:
    gt = generate(small_cfg.with_overrides(n_accounts=1500)).ground_truth
    # organic has lift 0 in default.yaml: removing it can only matter through MC noise
    assert gt.share["organic"] < 0.03
    assert gt.share["webinar"] > gt.share["paid"]


def test_raw_tables_carry_realistic_noise(small_cfg: SimConfig) -> None:
    t = generate(small_cfg.with_overrides(n_accounts=1000)).tables
    contacts = t["contacts"]
    assert contacts["account_id"].isna().mean() > 0.1  # unlinked CRM contacts
    assert (contacts["email"].str.contains("gmail")).any()
    assert t["identifies"]["anonymous_id"].nunique() < len(contacts)  # identity gaps
    assert set(t["email_events"]["event_type"]) >= {"sent", "click"}
    assert (t["activities"]["outcome"] == "no_answer").any()
    opps = t["opportunities"]
    assert set(opps["stage_name"]) <= {"open", "closed_won", "closed_lost"}
    assert (opps.loc[opps.is_won, "closed_at"].notna()).all()


def test_write_round_trip(tmp_path: object, small_cfg: SimConfig) -> None:
    from pathlib import Path

    import duckdb

    out = Path(str(tmp_path))
    data = generate(small_cfg)
    data.write(out)
    con = duckdb.connect()
    n = con.execute(f"select count(*) from read_parquet('{out / 'accounts.parquet'}')").fetchone()
    assert n is not None and n[0] == small_cfg.n_accounts
    assert (out / "ground_truth.json").exists()
