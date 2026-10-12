"""The no-leakage tests (ADR-0001): features agree with the naive as-of computation, and data after t never
changes a feature at t."""

import numpy as np
import pandas as pd
import pytest

from airalert.alarm_signal import classify
from airalert.features import HORIZONS, UNIT_FEATURES, build_grid, reference_features
from airalert.labels import merge_per_unit, unit_alerts_from_messages
from airalert.units import forecast_unit_ids

START = pd.Timestamp("2026-08-01 00:00", tz="UTC")
END = pd.Timestamp("2026-08-08 00:00", tz="UTC")
DATA_END = pd.Timestamp("2026-08-10 00:00", tz="UTC")
UNITS = ("A/a", "A/b", "B/c", "B/d")  # B/d has no alerts until the future is rewritten


def random_periods(seed: int, units=("A/a", "A/b", "B/c"), n=60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in units:
        t = START - pd.Timedelta(days=9)
        for _ in range(n):
            t += pd.Timedelta(minutes=int(rng.integers(5, 600)))
            end = t + pd.Timedelta(minutes=int(rng.integers(1, 400)), seconds=int(rng.integers(0, 60)))
            rows.append((unit, t, end))
            t = end
    return pd.DataFrame(rows, columns=["unit_id", "started_at", "finished_at"])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fast_features_match_the_naive_as_of_computation(seed):
    merged = random_periods(seed)
    grid = build_grid(merged, START, END, DATA_END, UNITS)
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(grid), size=60, replace=False):
        row = grid.iloc[i]
        expected = reference_features(merged, row.unit_id, row.t)
        for name in UNIT_FEATURES:
            assert row[name] == pytest.approx(expected[name], abs=1e-6), (name, row.unit_id, row.t)


def change_everything_after(merged: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Rewrite the future: alerts that started after `cutoff` are replaced, and ends after it are moved."""
    rng = np.random.default_rng(seed)
    past = merged[merged.started_at < cutoff].copy()
    later_end = past.finished_at >= cutoff
    past.loc[later_end, "finished_at"] = cutoff + pd.to_timedelta(rng.integers(1, 2000, later_end.sum()), unit="min")
    future = random_periods(seed + 100, units=UNITS)  # includes the first ever alerts of unit B/d
    future = future[future.started_at > past.finished_at.max() + pd.Timedelta(minutes=1)]
    return pd.concat([past, future], ignore_index=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_data_after_t_never_changes_features_at_t(seed):
    merged = random_periods(seed)
    cutoff = pd.Timestamp("2026-08-05 13:00", tz="UTC")
    original = build_grid(merged, START, END, DATA_END, UNITS)
    rewritten = build_grid(change_everything_after(merged, cutoff, seed), START, END, DATA_END, UNITS)
    feature_columns = [c for c in original.columns if not c.startswith(("label_", "published_label_"))]
    assert rewritten.loc[rewritten.t > cutoff + pd.Timedelta(hours=24)].query("unit_id == 'B/d'").on_alert.any()
    known = original.t <= cutoff
    pd.testing.assert_frame_equal(original.loc[known, feature_columns].reset_index(drop=True),
                                  rewritten.loc[known, feature_columns].reset_index(drop=True))
    assert not original.loc[known, "label_h24"].equals(rewritten.loc[known, "label_h24"])  # labels do look ahead


def test_labels_any_overlap_and_missing_after_data_end():
    merged = pd.DataFrame({"unit_id": ["A/a"], "started_at": [pd.Timestamp("2026-08-03 10:59:30", tz="UTC")],
                           "finished_at": [pd.Timestamp("2026-08-03 11:00:10", tz="UTC")]})
    grid = build_grid(merged, START, END, pd.Timestamp("2026-08-07 12:00", tz="UTC"), ["A/a"]).set_index("t")
    assert grid.loc["2026-08-03 10:00", "label_h1"] == 1  # 30 s overlap counts
    assert grid.loc["2026-08-03 11:00", "label_h1"] == 1
    assert grid.loc["2026-08-03 12:00", "label_h1"] == 0
    assert np.isnan(grid.loc["2026-08-07 12:00", "label_h1"])  # that hour is after the data ends
    assert set(f"label_h{h}" for h in HORIZONS) <= set(grid.columns)


def test_neighbour_share_uses_shared_borders_and_known_state():
    kyiv_neighbours = ["Київська область/Бориспільський район", "Київська область/Броварський район"]
    merged = pd.DataFrame({
        "unit_id": ["м. Київ", *kyiv_neighbours],
        "started_at": pd.to_datetime(["2026-08-02 00:00", "2026-08-03 09:30", "2026-08-03 11:30"], utc=True),
        "finished_at": pd.to_datetime(["2026-08-02 01:00", "2026-08-03 12:30", "2026-08-03 12:30"], utc=True),
    })
    grid = build_grid(merged, START, END, DATA_END, forecast_unit_ids())
    kyiv = grid[grid.unit_id == "м. Київ"].set_index("t")
    assert kyiv.loc["2026-08-03 10:00", "neighbour_on_alert_share"] == pytest.approx(1 / 6)  # one of its six neighbours
    assert kyiv.loc["2026-08-03 12:00", "neighbour_on_alert_share"] == pytest.approx(2 / 6)
    assert kyiv.loc["2026-08-03 12:00", "neighbour_starts_last_1h"] == 1
    assert kyiv.loc["2026-08-03 13:00", "neighbour_on_alert_share"] == 0.0


def test_grid_covers_the_fixed_unit_list_and_rejects_others():
    merged = random_periods(1, units=("A/a",))
    grid = build_grid(merged, START, END, DATA_END, UNITS)
    assert set(grid.unit_id) == set(UNITS)  # calm units included
    assert (grid[grid.unit_id == "B/d"].on_alert == 0).all()
    with pytest.raises(ValueError):
        build_grid(merged, START, END, DATA_END, ["A/b"])


KROPYVNYTSKYI = "Кіровоградська область/Кропивницький район"


def grid_from_messages(*items):
    """Feature and label periods from channel messages, as labels.build makes them, then the grid."""
    messages = [{"id": i, "date": d, "text": t} for i, (d, t) in enumerate(items)]
    alerts, _, _ = unit_alerts_from_messages(messages, classify)
    periods = merge_per_unit(alerts, DATA_END)
    label_periods = merge_per_unit(alerts[~alerts.retracted], DATA_END)
    return build_grid(periods, START, END, DATA_END, forecast_unit_ids(), label_periods)


def test_a_later_retraction_changes_labels_but_no_feature_before_it():
    start = ("2026-08-03T16:50:00+00:00", "🚨 Кропивницький район (Кіровоградська обл.)\nПовітряна тривога. Прямуйте в укриття!")
    retraction = ("2026-08-03T17:40:00+00:00", "🟢 Кропивницький район (Кіровоградська обл.)\n"
                  "Просимо вибачення, сталась технічна помилка. Тривога в області відсутня.")
    all_clear = ("2026-08-03T17:40:00+00:00", "🟢 Кропивницький район (Кіровоградська обл.)\nВідбій тривоги. Будьте обережні!")
    retracted, cleared = grid_from_messages(start, retraction), grid_from_messages(start, all_clear)
    features = [c for c in retracted.columns if not c.startswith(("label_", "published_label_"))]
    known = retracted.t <= pd.Timestamp("2026-08-03 17:00", tz="UTC")  # the retraction came at 17:40
    pd.testing.assert_frame_equal(retracted.loc[known, features].reset_index(drop=True),
                                  cleared.loc[known, features].reset_index(drop=True))
    at = lambda grid, t: grid[(grid.unit_id == KROPYVNYTSKYI) & (grid.t == pd.Timestamp(t, tz="UTC"))].iloc[0]  # noqa: E731
    assert at(retracted, "2026-08-03 17:00").on_alert == 1  # believed on alert at 17:00
    assert at(retracted, "2026-08-03 16:00").label_h1 == 0 and at(cleared, "2026-08-03 16:00").label_h1 == 1
