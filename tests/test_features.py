"""The no-leakage tests (ADR-0001): features agree with the naive as-of computation, and data after t never
changes a feature at t."""

import numpy as np
import pandas as pd
import pytest

from airalert.features import HORIZONS, UNIT_FEATURES, build_grid, reference_features

START = pd.Timestamp("2026-08-01 00:00", tz="UTC")
END = pd.Timestamp("2026-08-08 00:00", tz="UTC")
DATA_END = pd.Timestamp("2026-08-10 00:00", tz="UTC")


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
    grid = build_grid(merged, START, END, DATA_END)
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
    future = random_periods(seed + 100)
    future = future[future.started_at > past.finished_at.max() + pd.Timedelta(minutes=1)]
    return pd.concat([past, future], ignore_index=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_data_after_t_never_changes_features_at_t(seed):
    merged = random_periods(seed)
    cutoff = pd.Timestamp("2026-08-05 13:00", tz="UTC")
    feature_columns = [c for c in build_grid(merged, START, END, DATA_END).columns if not c.startswith("label_")]
    original = build_grid(merged, START, END, DATA_END)
    rewritten = build_grid(change_everything_after(merged, cutoff, seed), START, END, DATA_END)
    known = original.t <= cutoff
    pd.testing.assert_frame_equal(original.loc[known, feature_columns].reset_index(drop=True),
                                  rewritten.loc[known, feature_columns].reset_index(drop=True))
    assert not original.loc[known, "label_h24"].equals(rewritten.loc[known, "label_h24"])  # labels do look ahead


def test_labels_any_overlap_and_missing_after_data_end():
    merged = pd.DataFrame({"unit_id": ["A/a"], "started_at": [pd.Timestamp("2026-08-03 10:59:30", tz="UTC")],
                           "finished_at": [pd.Timestamp("2026-08-03 11:00:10", tz="UTC")]})
    grid = build_grid(merged, START, END, pd.Timestamp("2026-08-07 12:00", tz="UTC")).set_index("t")
    assert grid.loc["2026-08-03 10:00", "label_h1"] == 1  # 30 s overlap counts
    assert grid.loc["2026-08-03 11:00", "label_h1"] == 1
    assert grid.loc["2026-08-03 12:00", "label_h1"] == 0
    assert np.isnan(grid.loc["2026-08-07 12:00", "label_h1"])  # that hour is after the data ends
    assert set(f"label_h{h}" for h in HORIZONS) <= set(grid.columns)
