"""Features and labels on the hourly grid (ADR-0001, DESIGN.md section 3).

One row per (unit, forecast time t), where t is a whole hour. Every feature uses only what was known
strictly before t:
- an alert's start is known from its start time; its end only from its end time, so an alert still on at t
  counts as ongoing, however long it later lasted;
- per-hour aggregates are taken only over hours that ended by t.
Labels look forward: `label_h{h}` is 1 if the unit is on alert at any moment of the hour starting at
t + (h - 1) hours (Q7: any overlap counts). Labels whose hour ends after the data ends are missing.
Features read the alert periods as published; labels read the corrected ones (retracted alerts dropped).

`reference_features` computes the same alert-history features naively, straight from the alert periods
filtered by what was known at t. Tests check that both agree and that adding or changing data after t never
changes a feature at t.
"""

import pathlib

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


def _unit_block(timeline: UnitTimeline, t: np.ndarray, known_active: np.ndarray, label_active: np.ndarray,
                corrected_at: np.ndarray, k0: int) -> dict:
    """`known_active` / `label_active`: per grid hour, was the unit on alert, from feature (published) / label
    (corrected) periods. `corrected_at`: per grid hour, when the last correction changing it was posted (-1: none)."""
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
    k = np.arange(len(t)) + k0  # index of the hour starting at t in the per-hour arrays
    for h in HORIZONS:
        target = k + h - 1
        inside = target < len(label_active)
        clipped = np.minimum(target, len(label_active) - 1)
        out[f"label_h{h}"] = np.where(inside, label_active[clipped], np.nan)
        out[f"published_label_h{h}"] = np.where(inside, known_active[clipped], np.nan)  # as known before any correction
        out[f"label_corrected_at_h{h}"] = pd.to_datetime(np.where(inside & (corrected_at[clipped] >= 0), corrected_at[clipped], np.nan),
                                                          unit="s", utc=True)
        for lag, name in ((24, "day"), (168, "week")):  # was the target hour on alert a day / a week earlier?
            out[f"target_active_{name}_ago_h{h}"] = known_active[target - lag]  # an hour that ended by t
    return out


def build_grid(periods: pd.DataFrame, start, end, data_end, units, label_periods: pd.DataFrame | None = None,
               corrections: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hourly rows for t in [start, end) for every unit in `units` (calm ones included), whether or not it has periods.

    `periods` (columns unit_id, started_at, finished_at) are the alert periods as published, which features use;
    `label_periods` the corrected truth, which labels use (default: the same). `corrections` are the retracted alerts
    (finished_at = when the retraction was posted). For each horizon the grid also holds the label as published and
    when a correction changed it (`published_label_h*`, `label_corrected_at_h*`), so that training at a cutoff can
    use the label as it was known then (ADR-0007). The unit list is fixed in advance (units.forecast_unit_ids()),
    so a unit's first alert never changes country, oblast or neighbour features before it.
    """
    start, end, data_end = (pd.Timestamp(x).tz_convert("UTC") for x in (start, end, data_end))
    label_periods = periods if label_periods is None else label_periods
    units = sorted(units)
    unknown = (set(periods.unit_id) | set(label_periods.unit_id)) - set(units)
    if unknown:
        raise ValueError(f"periods for units outside the unit list: {sorted(unknown)[:5]}")
    hours = pd.date_range(start - 168 * HOUR, end + 24 * HOUR, freq="h", inclusive="left")  # with lookback and lookahead
    k0 = 168
    edges = _seconds(hours.append(pd.DatetimeIndex([hours[-1] + HOUR])))
    t = edges[k0:k0 + int((end - start) / HOUR)]
    complete = edges[1:] <= _seconds([data_end])[0]  # hours that ended before the data did

    def hourly(timeline):
        return np.where(complete, (np.diff(timeline.alert_seconds_before(edges)) > 0).astype(float), np.nan)

    def correction_times(group):
        """Per grid hour: the latest posting time of a retraction whose alert overlapped that hour (-1: none)."""
        out = np.full(len(edges) - 1, -1, dtype=np.int64)
        for s_, e_ in zip(_seconds(group.started_at), _seconds(group.finished_at)):
            hit = (edges[:-1] < e_) & (edges[1:] > s_)
            out[hit] = np.maximum(out[hit], e_)
        return out

    corrections = periods.iloc[:0] if corrections is None else corrections
    by_unit = dict(tuple(periods.groupby("unit_id")))
    labels_by_unit = dict(tuple(label_periods.groupby("unit_id")))
    corrections_by_unit = dict(tuple(corrections.groupby("unit_id")))
    empty = periods.iloc[:0]
    blocks = []
    for unit_id in units:
        group = by_unit.get(unit_id, empty)
        label_group = labels_by_unit.get(unit_id, empty)
        timeline = UnitTimeline(group.started_at, group.finished_at)
        label_timeline = timeline if label_periods is periods else UnitTimeline(label_group.started_at, label_group.finished_at)
        block = _unit_block(timeline, t, hourly(timeline), hourly(label_timeline),
                            correction_times(corrections_by_unit.get(unit_id, empty)), k0)
        block["unit_id"] = unit_id
        block["t"] = pd.to_datetime(t, unit="s", utc=True)
        blocks.append(pd.DataFrame(block))
    grid = pd.concat(blocks, ignore_index=True)
    return _add_cross_unit_and_calendar(grid)


NEIGHBOURS = pathlib.Path(__file__).resolve().parent / "reference" / "neighbours.csv"


def _add_neighbours(grid: pd.DataFrame) -> pd.DataFrame:
    """Share of a unit's neighbours (shared border, Q11) on alert at t, and their alert starts in the last hour."""
    pairs = pd.read_csv(NEIGHBOURS)
    units = pd.Index(sorted(grid["unit_id"].unique()))
    pairs = pairs[pairs.unit_id.isin(units) & pairs.neighbour_id.isin(units)]
    adjacency = np.zeros((len(units), len(units)))
    adjacency[units.get_indexer(pairs.unit_id), units.get_indexer(pairs.neighbour_id)] = 1
    degree = adjacency.sum(axis=1, keepdims=True)
    for column, name, normalise in (("on_alert", "neighbour_on_alert_share", True), ("starts_last_1h", "neighbour_starts_last_1h", False)):
        wide = grid.pivot(index="t", columns="unit_id", values=column).reindex(columns=units).fillna(0).to_numpy()
        weights = np.divide(adjacency, degree, out=np.zeros_like(adjacency), where=degree > 0) if normalise else adjacency
        values = pd.DataFrame(wide @ weights.T, index=grid["t"].drop_duplicates().sort_values().to_numpy(), columns=units)
        long = values.stack().rename(name).reset_index().rename(columns={"level_0": "t", "level_1": "unit_id"})
        long.columns = ["t", "unit_id", name]
        grid = grid.merge(long, on=["t", "unit_id"], how="left")
    return grid


def _add_cross_unit_and_calendar(grid: pd.DataFrame) -> pd.DataFrame:
    grid = _add_neighbours(grid)
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
