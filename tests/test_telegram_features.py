"""Telegram features must only use posts available before t (ADR-0001)."""

import numpy as np
import pandas as pd
import pytest

from airalert.features import build_grid
from airalert.telegram_features import ALL_TELEGRAM_FEATURES, add_telegram_features, reference_counts

START, END = pd.Timestamp("2026-08-01", tz="UTC"), pd.Timestamp("2026-08-04", tz="UTC")
DATA_END = pd.Timestamp("2026-08-06", tz="UTC")
KYIV, BORYSPIL = "м. Київ", "Київська область/Бориспільський район"
UNITS = [KYIV, BORYSPIL, "Сумська область/Сумський район"]


def grid():
    no_alerts = pd.DataFrame(columns=["unit_id", "started_at", "finished_at"]).astype({"started_at": "datetime64[ns, UTC]",
                                                                                      "finished_at": "datetime64[ns, UTC]"})
    return build_grid(no_alerts, START, END, DATA_END, UNITS)


def event(post_id, minute_offset, kind, scope, unit=None, base="2026-08-02 10:00", channel="kpszsu"):
    return {"post_id": post_id, "channel": channel, "available_at": pd.Timestamp(base, tz="UTC") + pd.Timedelta(minutes=minute_offset),
            "type": kind, "scope": scope, "unit_id": unit}


def random_events(seed, n=400):
    rng = np.random.default_rng(seed)
    kinds = ["drones", "jet_drones", "missiles", "ballistic", "guided_bombs", "aviation", "intercept", "clear"]
    rows = []
    for i in range(n):  # both channels reuse the same post ids, as the real ones do
        minute = int(rng.integers(-60 * 30, 60 * 60))
        kind, channel = kinds[rng.integers(len(kinds))], ["kpszsu", "deraketaua"][rng.integers(2)]
        post_id = int(rng.integers(n // 2))
        rows.append(event(post_id, minute, kind, "national", channel=channel))
        unit = UNITS[rng.integers(len(UNITS))]
        rows.append(event(post_id, minute, kind, ["local", "oblast"][rng.integers(2)], unit, channel=channel))
    return pd.DataFrame(rows)


def at(frame, unit, time):
    return frame[(frame.unit_id == unit) & (frame.t == pd.Timestamp(time, tz="UTC"))].iloc[0]


def test_counts_use_only_posts_before_t_and_group_types():
    events = pd.DataFrame([
        event(1, -50, "jet_drones", "local", KYIV), event(1, -50, "recon", "local", KYIV),  # one post, one count
        event(2, 0, "drones", "local", KYIV),  # exactly at t=10:00: not yet known
        event(3, -200, "drones", "local", KYIV),  # outside 180 min
        event(1, -50, "jet_drones", "national"), event(2, 0, "drones", "national"),
    ])
    row = at(add_telegram_features(grid(), events), KYIV, "2026-08-02 10:00")
    assert (row.tg_local_drones_15m, row.tg_local_drones_60m, row.tg_local_drones_180m) == (0, 1, 1)
    assert row.tg_national_drones_60m == 1 and row.tg_minutes_since_local_threat == 50
    assert row.tg_neighbour_threat_60m == 0
    assert row.tg_local_threat_60m == 1  # post 1: jet drone and recon, one post
    assert at(add_telegram_features(grid(), events), BORYSPIL, "2026-08-02 10:00").tg_neighbour_threat_60m == 1  # Kyiv borders Boryspil


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fast_counts_match_the_naive_computation(seed):
    events = random_events(seed)
    frame = add_telegram_features(grid(), events)
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(frame), size=40, replace=False):
        row = frame.iloc[i]
        for scope, group, w in (("local", "drones", 60), ("oblast", "missiles", 180), ("local", "clear", 15),
                                ("local", "threat", 60), ("oblast", "threat", 60), ("national", "drones", 180),
                                ("national", "aviation", 360)):
            assert row[f"tg_{scope}_{group}_{w}m"] == reference_counts(events, row.unit_id, row.t, scope, group, w)


@pytest.mark.parametrize("seed", [1, 2])
def test_posts_after_t_never_change_features_at_t(seed):
    events = random_events(seed)
    cutoff = pd.Timestamp("2026-08-02 13:00", tz="UTC")
    rewritten = pd.concat([events[events.available_at < cutoff], random_events(seed + 50).assign(
        available_at=lambda e: cutoff + (e.available_at - e.available_at.min()), post_id=lambda e: e.post_id + 10_000)])
    a, b = add_telegram_features(grid(), events), add_telegram_features(grid(), rewritten)
    known = a.t <= cutoff
    pd.testing.assert_frame_equal(a.loc[known, ALL_TELEGRAM_FEATURES].reset_index(drop=True),
                                  b.loc[known, ALL_TELEGRAM_FEATURES].reset_index(drop=True))


def test_a_post_edited_after_t_is_not_seen_before_its_edit():
    """threats.py dates a post by its edit time; an edit at 10:30 makes it invisible at 10:00, visible at 11:00."""
    events = pd.DataFrame([event(1, 30, "drones", "local", KYIV)])
    frame = add_telegram_features(grid(), events)
    assert at(frame, KYIV, "2026-08-02 10:00").tg_local_drones_60m == 0
    assert at(frame, KYIV, "2026-08-02 11:00").tg_local_drones_60m == 1


def test_the_same_post_id_in_two_channels_is_two_posts():
    events = pd.DataFrame([event(7, -30, "drones", "national"), event(7, -20, "drones", "national", channel="deraketaua"),
                           event(7, -30, "drones", "local", KYIV), event(7, -20, "drones", "local", KYIV, channel="deraketaua")])
    row = at(add_telegram_features(grid(), events), KYIV, "2026-08-02 10:00")
    assert row.tg_national_drones_60m == 2 and row.tg_local_drones_60m == 2 and row.tg_local_threat_60m == 2


def test_a_unit_we_do_not_forecast_changes_no_feature():
    """Luhansk raions are not forecast; their posts must not land in another unit's column."""
    base = pd.DataFrame([event(1, -50, "drones", "local", BORYSPIL)])
    extra = pd.concat([base, pd.DataFrame([event(2, -5, "drones", "local", "Луганська область/Сватівський район")])])
    a, b = add_telegram_features(grid(), base), add_telegram_features(grid(), extra)
    local = [c for c in ALL_TELEGRAM_FEATURES if not c.startswith("tg_national")]
    pd.testing.assert_frame_equal(a[local], b[local])
