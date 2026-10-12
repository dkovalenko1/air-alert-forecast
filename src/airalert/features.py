"""Features and labels on the hourly grid (ADR-0001, DESIGN.md section 3).

One row per (unit, forecast time t), where t is a whole hour. Every feature uses only what was known
strictly before t:
- an alert's start is known from its start time; its end only from its end time, so an alert still on at t
  counts as ongoing, however long it later lasted;
- per-hour aggregates are taken only over hours that ended by t.
Labels look forward: `label_h{h}` is 1 if the unit is on alert at any moment of the hour starting at
t + (h - 1) hours (Q7: any overlap counts). Labels whose hour ends after the data ends are missing.

`reference_features` computes the same alert-history features naively, straight from the alert periods
filtered by what was known at t. Tests check that both agree and that adding or changing data after t never
changes a feature at t.
"""

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 6, 12, 24)  # Q10
KYIV = "Europe/Kyiv"
HOUR = pd.Timedelta(hours=1)
LOOKBACKS_H = (1, 3, 6, 24)  # for counting alert starts
SHARE_WINDOWS_H = (24, 168)  # for the share of time on alert
CAP_MINUTES = 7 * 24 * 60  # "minutes since" features are capped at a week

UNIT_FEATURES = (
    ["on_alert", "alert_minutes_so_far", "minutes_since_all_clear"]
    + [f"starts_last_{h}h" for h in LOOKBACKS_H]
    + [f"alert_share_last_{h}h" for h in SHARE_WINDOWS_H]
)


def _seconds(times) -> np.ndarray:
    return pd.DatetimeIndex(times).as_unit("s").asi8.astype(np.int64)


def _merge(s: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of periods sorted by start. Merging never moves a start later, so it is safe for as-of queries."""
    merged_s, merged_e = [], []
    for start, end in zip(s, e):
        if merged_e and start <= merged_e[-1]:
            merged_e[-1] = max(merged_e[-1], end)
        else:
            merged_s.append(start)
            merged_e.append(end)
    return np.array(merged_s, dtype=np.int64), np.array(merged_e, dtype=np.int64)


class UnitTimeline:
    """One unit's alert periods (overlaps merged), answering questions about any set of times."""

    def __init__(self, starts, ends):
        order = np.argsort(_seconds(starts), kind="stable")
        self.s, self.e = _merge(_seconds(starts)[order], _seconds(ends)[order])
        self.done = np.concatenate([[0], np.cumsum(self.e - self.s)])  # total alert seconds of periods before i

    def alert_seconds_before(self, x: np.ndarray) -> np.ndarray:
        """Seconds on alert in (-inf, x), counting an ongoing alert only up to x."""
        finished = np.searchsorted(self.e, x, side="right")  # periods that ended by x
        partial = np.zeros_like(x)
        has_next = finished < len(self.s)
        idx = np.minimum(finished, len(self.s) - 1)
        if len(self.s):
            partial = np.where(has_next & (self.s[idx] < x), x - self.s[idx], 0)
        return self.done[finished] + partial

    def state_at(self, x: np.ndarray):
        """For each x: on alert (all-clear not yet posted before x), start of that alert, end of the last finished one."""
        last = np.searchsorted(self.s, x, side="left") - 1  # last period that started strictly before x
        valid = last >= 0
        idx = np.maximum(last, 0)
        on = valid & (self.e[idx] >= x) if len(self.s) else np.zeros_like(x, dtype=bool)
        start = np.where(on, self.s[idx] if len(self.s) else 0, -1)
        last_end = np.where(valid & ~on, self.e[idx] if len(self.s) else 0, -1)
        return on, start, last_end

    def starts_between(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.s, b, side="left") - np.searchsorted(self.s, a, side="left")


def _unit_block(timeline: UnitTimeline, t: np.ndarray, slot_active: np.ndarray, k0: int) -> dict:
    on, start, last_end = timeline.state_at(t)
    out = {
        "on_alert": on.astype(np.int8),
        "alert_minutes_so_far": np.where(on, (t - start) / 60, 0.0),
        "minutes_since_all_clear": np.where(on, 0.0, np.where(last_end >= 0, np.minimum((t - last_end) / 60, CAP_MINUTES), CAP_MINUTES)),
    }
    for h in LOOKBACKS_H:
        out[f"starts_last_{h}h"] = timeline.starts_between(t - h * 3600, t)
    before_t = timeline.alert_seconds_before(t)
    for h in SHARE_WINDOWS_H:
        out[f"alert_share_last_{h}h"] = (before_t - timeline.alert_seconds_before(t - h * 3600)) / (h * 3600)
    k = np.arange(len(t)) + k0  # index of the hour starting at t in `slot_active`
    for h in HORIZONS:
        target = k + h - 1
        out[f"label_h{h}"] = slot_active[target] if target.max() < len(slot_active) else np.where(
            target < len(slot_active), slot_active[np.minimum(target, len(slot_active) - 1)], np.nan)
        for lag, name in ((24, "day"), (168, "week")):  # was the target hour on alert a day / a week earlier?
            out[f"target_active_{name}_ago_h{h}"] = slot_active[target - lag]
    return out


def build_grid(merged: pd.DataFrame, start, end, data_end) -> pd.DataFrame:
    """Hourly rows for t in [start, end) for every unit in `merged` (columns unit_id, started_at, finished_at)."""
    start, end, data_end = (pd.Timestamp(x).tz_convert("UTC") for x in (start, end, data_end))
    hours = pd.date_range(start - 168 * HOUR, end + 24 * HOUR, freq="h", inclusive="left")  # with lookback and lookahead
    k0 = 168
    edges = _seconds(hours.append(pd.DatetimeIndex([hours[-1] + HOUR])))
    t = edges[k0:k0 + int((end - start) / HOUR)]
    complete = edges[1:] <= _seconds([data_end])[0]  # hours that ended before the data did

    blocks = []
    for unit_id, group in merged.sort_values("started_at").groupby("unit_id", sort=True):
        timeline = UnitTimeline(group.started_at, group.finished_at)
        seconds = timeline.alert_seconds_before(edges)
        slot_active = np.where(complete, (np.diff(seconds) > 0).astype(float), np.nan)
        block = _unit_block(timeline, t, slot_active, k0)
        block["unit_id"] = unit_id
        block["t"] = pd.to_datetime(t, unit="s", utc=True)
        blocks.append(pd.DataFrame(block))
    grid = pd.concat(blocks, ignore_index=True)
    return _add_cross_unit_and_calendar(grid)


def _add_cross_unit_and_calendar(grid: pd.DataFrame) -> pd.DataFrame:
    grid["oblast"] = grid["unit_id"].str.split("/").str[0].astype("category")
    grid["unit_id"] = grid["unit_id"].astype("category")
    by_t = grid.groupby("t")
    grid["national_on_alert_share"] = by_t["on_alert"].transform("mean")
    grid["national_starts_last_1h"] = by_t["starts_last_1h"].transform("sum")
    by_oblast = grid.groupby(["t", "oblast"])
    grid["oblast_on_alert_share"] = by_oblast["on_alert"].transform("mean")  # same-oblast feature (Q11)
    local = grid["t"].dt.tz_convert(KYIV)
    grid["hour"], grid["weekday"], grid["month"] = local.dt.hour, local.dt.weekday, local.dt.month
    for h in HORIZONS:
        target = (grid["t"] + (h - 1) * HOUR).dt.tz_convert(KYIV)
        grid[f"target_hour_h{h}"], grid[f"target_weekday_h{h}"] = target.dt.hour, target.dt.weekday
    return grid


def reference_features(merged: pd.DataFrame, unit_id: str, t: pd.Timestamp) -> dict:
    """The alert-history features for one unit at one time, computed naively from what was known before t."""
    periods = merged[(merged.unit_id == unit_id) & (merged.started_at < t)]  # starts known before t
    ongoing = periods[periods.finished_at >= t]  # no all-clear before t
    ended = periods[periods.finished_at < t]
    on = len(ongoing) > 0
    out = {
        "on_alert": int(on),
        "alert_minutes_so_far": (t - ongoing.started_at.min()).total_seconds() / 60 if on else 0.0,
        "minutes_since_all_clear": 0.0 if on else min((t - ended.finished_at.max()).total_seconds() / 60, CAP_MINUTES)
        if len(ended) else float(CAP_MINUTES),
    }
    for h in LOOKBACKS_H:
        out[f"starts_last_{h}h"] = int((periods.started_at >= t - h * HOUR).sum())
    for h in SHARE_WINDOWS_H:
        window_start = t - h * HOUR
        clipped_end = periods.finished_at.where(periods.finished_at < t, t)  # an ongoing alert counts only up to t
        overlap = (clipped_end.clip(lower=window_start) - periods.started_at.clip(lower=window_start)).dt.total_seconds()
        out[f"alert_share_last_{h}h"] = overlap.clip(lower=0).sum() / (h * 3600)
    return out
