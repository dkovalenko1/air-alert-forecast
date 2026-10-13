"""Label-quality checks: our label source against independent records (Q20, Q2, ADR-0006).

Alert starts and all-clears are compared per unit: an event in one source counts as found if the other source
has the same kind of event for the same unit within 5 minutes. Both directions are reported per Kyiv calendar
month, against the pass mark fixed in advance (Q20): at least 98% each way, every month. Every month × kind ×
direction of the requested window gets a row; one with no events is marked as missing coverage and fails.
Matching looks at the other source's events beyond the window too, so an event near the edge is not lost.
Outages of the reference (no message at all for over 24 h; normal quiet spells last at most about 12 h) are left
out on both sides: the official channel was silent from 2026-08-29 07:55 to 2026-09-02 16:48 Kyiv time.

- Q20: `@UkraineAlarmSignal` vs the official channel `@air_alert_ua`, 2023-01 to 2026-08.
- Q2: `@UkraineAlarmSignal` vs alerts.in.ua over the test period (the official channel is silent there).
"""

import glob
import json

import numpy as np
import pandas as pd

from airalert.alarm_signal import Location
from airalert.settings import RAW_DIR
from airalert.units import UnknownPlace, resolve, units

PASS_MARK = 0.98
TOLERANCE = pd.Timedelta(minutes=5)
OUTAGE = pd.Timedelta(hours=24)


def outages(message_times, min_silence=OUTAGE) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Windows in which a channel posted nothing for longer than `min_silence`: (last message, next message)."""
    times = pd.DatetimeIndex(sorted(pd.to_datetime(list(message_times), utc=True)))
    gaps = times[1:] - times[:-1]
    return [(a, b) for a, b, gap in zip(times[:-1], times[1:], gaps) if gap > min_silence]


def events(alerts: pd.DataFrame) -> pd.DataFrame:
    """Start and all-clear events per unit, one row per distinct (unit, kind, time). Retracted alerts are left out."""
    if "retracted" in alerts:
        alerts = alerts[~alerts.retracted.astype(bool)]
    announced = alerts[~alerts.missed_start.astype(bool)] if "missed_start" in alerts else alerts
    # a never-announced alert (missed start) has an all-clear event but no start event
    starts = announced[["unit_id", "started_at"]].rename(columns={"started_at": "time"}).assign(kind="start")
    ends = alerts.dropna(subset=["finished_at"])[["unit_id", "finished_at"]].rename(columns={"finished_at": "time"}).assign(kind="end")
    both = pd.concat([starts, ends]).drop_duplicates()
    both["time"] = pd.to_datetime(both["time"], utc=True)  # also when there are no events at all
    return both.sort_values("time", ignore_index=True)


def found_share(source: pd.DataFrame, other: pd.DataFrame, tolerance=TOLERANCE) -> pd.Series:
    """For each event in `source`: is there an event of the same kind and unit in `other` within `tolerance`?"""
    found = np.zeros(len(source), dtype=bool)
    if source.empty:
        return pd.Series(found, index=source.index)
    for (unit_id, kind), group in source.groupby(["unit_id", "kind"]):
        times = np.sort(other.loc[(other.unit_id == unit_id) & (other.kind == kind), "time"].to_numpy())
        if not len(times):
            continue
        x = group.time.to_numpy()
        i = np.clip(np.searchsorted(times, x), 1, len(times)) - 1
        nearest = np.minimum(np.abs(times[i] - x), np.abs(times[np.minimum(i + 1, len(times) - 1)] - x))
        found[source.index.get_indexer(group.index)] = nearest <= tolerance.to_timedelta64()
    return pd.Series(found, index=source.index)


def _kyiv(x) -> pd.Timestamp:
    """A window bound in Kyiv time: dates without a time zone are Kyiv calendar dates, others are converted."""
    x = pd.Timestamp(x)
    return x.tz_localize("Europe/Kyiv") if x.tzinfo is None else x.tz_convert("Europe/Kyiv")


def monthly_report(labels: pd.DataFrame, reference: pd.DataFrame, start, end, exclude=()) -> pd.DataFrame:
    """Per Kyiv month and event kind: share of reference events found in the labels, and the other way round.
    Months of [start, end) with no events of a kind get a row with `events` 0, `found` NaN and `passes` False.
    Events within `exclude` windows (reference outages), widened by the tolerance, are left out on both sides."""
    start, end = _kyiv(start), _kyiv(end)
    all_ours, all_theirs = events(labels), events(reference)

    def window(ev):
        keep = (ev.time >= start) & (ev.time < end)
        for a, b in exclude:
            keep &= ~((ev.time >= a - TOLERANCE) & (ev.time <= b + TOLERANCE))
        return ev[keep].reset_index(drop=True)

    ours, theirs = window(all_ours), window(all_theirs)
    theirs["found"] = found_share(theirs, all_ours).to_numpy()  # matched against everything, not just the window
    ours["found"] = found_share(ours, all_theirs).to_numpy()
    months = pd.period_range(start.tz_localize(None), (end - pd.Timedelta(seconds=1)).tz_localize(None), freq="M").strftime("%Y-%m")
    rows = []
    for direction, ev in (("reference found in labels", theirs), ("labels found in reference", ours)):
        month = ev.time.dt.tz_convert("Europe/Kyiv").dt.strftime("%Y-%m")
        stats = ev.groupby([month, ev.kind])["found"].agg(["size", "mean"])
        for m in months:
            for kind in ("start", "end"):
                n, share = stats.loc[(m, kind)] if (m, kind) in stats.index else (0, np.nan)
                rows.append({"month": m, "kind": kind, "direction": direction, "events": int(n), "found": share})
    report = pd.DataFrame(rows, columns=["month", "kind", "direction", "events", "found"])
    report["passes"] = (report.events > 0) & (report.found >= PASS_MARK)
    return report


def hourly_agreement(periods, label_periods, corrections, reference_periods, start, end, data_end, exclude=()) -> pd.DataFrame:
    """Per Kyiv month: does the Active label (any alert in the hour) agree with the reference's, raion-hour by
    raion-hour? This is what the model uses. Unknown label hours are counted separately and left out of the
    agreement; hours in `exclude` windows (reference outages) are left out entirely."""
    from airalert.features import build_grid
    from airalert.units import forecast_unit_ids

    units = forecast_unit_ids()
    ours = build_grid(periods, _kyiv(start), _kyiv(end), data_end, units, label_periods, corrections)
    theirs = build_grid(reference_periods[["unit_id", "started_at", "finished_at"]], _kyiv(start), _kyiv(end), data_end, units)
    y, ref = ours.label_h1.to_numpy(), theirs.label_h1.to_numpy()
    valid = ~np.isnan(ref)
    for a, b in exclude:
        valid &= ~((ours.t > a - pd.Timedelta(hours=1)) & (ours.t < b)).to_numpy()
    month = ours.t.dt.tz_convert("Europe/Kyiv").dt.strftime("%Y-%m").to_numpy()
    rows = []
    for m in sorted(set(month[valid])):
        sel = valid & (month == m)
        known = sel & ~np.isnan(y)
        rows.append({"month": m, "raion_hours": int(sel.sum()), "unknown_share": np.isnan(y[sel]).mean(),
                     "agree": np.mean(y[known] == ref[known]), "ours_alert_reference_calm": np.mean((y[known] == 1) & (ref[known] == 0)),
                     "ours_calm_reference_alert": np.mean((y[known] == 0) & (ref[known] == 1))})
    return pd.DataFrame(rows)


def alerts_in_ua_alerts() -> pd.DataFrame:
    """alerts.in.ua air raid alerts as unit alerts (same shape as labels.unit_alerts_from_messages).

    Raions map to themselves and oblasts to all their raions; Kyiv city (which the API lists as an oblast,
    "м. Київ") to its unit. Other city and hromada records are dropped, as in our labels (Q16). Snapshots overlap,
    so each alert id is taken from the latest snapshot that has it (a later one may add its all-clear).
    """
    excluded = {u.unit_id for u in units() if u.excluded}
    latest = {}
    for path in sorted(glob.glob(str(RAW_DIR / "alerts_in_ua" / "*" / "*.json"))):  # snapshot folders sort by time
        for alert in json.load(open(path, encoding="utf-8"))["alerts"]:
            if alert["alert_type"] == "air_raid":
                latest[alert["id"]] = alert
    return _alerts_in_ua_rows(latest.values(), excluded)


def _alerts_in_ua_rows(alerts, excluded=frozenset()) -> pd.DataFrame:
    rows = []
    for alert in alerts:
        level = {"raion": "raion", "oblast": "oblast", "city": "city"}.get(alert["location_type"])
        if level is None:
            continue
        oblast = alert.get("location_oblast")
        if oblast == "м. Київ":
            level, name, oblast = "city", "Київ", None
        else:
            name = alert["location_title"] if level != "oblast" else oblast
        try:
            unit_ids = resolve(Location(level, name, oblast))
        except UnknownPlace:
            continue
        for unit_id in unit_ids:
            if unit_id not in excluded:
                rows.append({"unit_id": unit_id, "started_at": alert["started_at"], "finished_at": alert["finished_at"]})
    frame = pd.DataFrame(rows, columns=["unit_id", "started_at", "finished_at"])
    for column in ("started_at", "finished_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def main() -> None:
    """Write reports/label_quality/official_by_month.csv (Q20) and alerts_in_ua_test_period.csv (Q2)."""
    from airalert.backtest import PERIODS
    from airalert.labels import INTERIM_DIR
    from airalert.settings import ROOT

    out = ROOT / "reports" / "label_quality"
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_alerts.parquet")
    official_path = INTERIM_DIR / "air_alert_ua_alerts.parquet"
    if official_path.exists():
        from airalert.labels import read_channel

        gaps = outages(m["date"] for m in read_channel("air_alert_ua"))
        for a, b in gaps:
            print(f"Official channel outage left out: {a.tz_convert('Europe/Kyiv'):%Y-%m-%d %H:%M} → {b.tz_convert('Europe/Kyiv'):%Y-%m-%d %H:%M} Kyiv time")
        report = monthly_report(labels, pd.read_parquet(official_path), "2023-01-01", "2026-09-01", exclude=gaps)
        report.to_csv(out / "official_by_month.csv", index=False)
        read = lambda name: pd.read_parquet(INTERIM_DIR / name)  # noqa: E731
        hourly = hourly_agreement(read("UkraineAlarmSignal_merged.parquet"), read("UkraineAlarmSignal_label_periods.parquet"),
                                  read("UkraineAlarmSignal_corrections.parquet"), read("air_alert_ua_label_periods.parquet"),
                                  "2023-01-01", "2026-09-08", read("air_alert_ua_label_periods.parquet").data_end.iloc[0], exclude=gaps)
        hourly.to_csv(out / "official_hourly_by_month.csv", index=False)
        print(f"Hour level (what the model uses): agreement on known raion-hours {hourly.agree.min():.2%} to {hourly.agree.max():.2%} "
              f"per month; unknown {hourly.unknown_share.min():.1%} to {hourly.unknown_share.max():.1%}")
        failing = report[~report.passes]
        print(f"Q20: {len(report)} month × kind × direction rows, {len(failing)} below the {PASS_MARK:.0%} pass mark")
        if len(failing):
            print(failing.sort_values("found").head(15).to_string(index=False))
    else:
        print("Q20 skipped: run `build-labels air_alert_ua` first.")
    start, end = PERIODS["test"]
    report = monthly_report(labels, alerts_in_ua_alerts(), start, end)
    report.to_csv(out / "alerts_in_ua_test_period.csv", index=False)
    print("Q2 (test period, vs alerts.in.ua):")
    print(report.to_string(index=False))
    from airalert.labels import merge_per_unit

    merged = pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_merged.parquet")
    data_end = merged.data_end.iloc[0]
    reference = merge_per_unit(alerts_in_ua_alerts(), data_end)
    hourly = hourly_agreement(merged, pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_label_periods.parquet"),
                              pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_corrections.parquet"), reference, start, end, data_end)
    hourly.to_csv(out / "alerts_in_ua_test_period_hourly.csv", index=False)
    known = hourly.raion_hours * (1 - hourly.unknown_share)
    print(f"Q2 hour level (what the model uses): agreement on known raion-hours "
          f"{(hourly.agree * known).sum() / known.sum():.2%} over the test period (months {hourly.agree.min():.2%} to "
          f"{hourly.agree.max():.2%}); unknown {(hourly.unknown_share * hourly.raion_hours).sum() / hourly.raion_hours.sum():.1%}")
