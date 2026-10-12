"""Label-quality checks: our label source against independent records (Q20, Q2, ADR-0006).

Alert starts and all-clears are compared per unit: an event in one source counts as found if the other source
has the same kind of event for the same unit within 5 minutes. Both directions are reported per Kyiv calendar
month, against the pass mark fixed in advance (Q20): at least 98% each way, every month. Every month × kind ×
direction of the requested window gets a row; one with no events is marked as missing coverage and fails.
Matching looks at the other source's events beyond the window too, so an event near the edge is not lost.

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


def events(alerts: pd.DataFrame) -> pd.DataFrame:
    """Start and all-clear events per unit, one row per distinct (unit, kind, time). Retracted alerts are left out."""
    if "retracted" in alerts:
        alerts = alerts[~alerts.retracted.astype(bool)]
    starts = alerts[["unit_id", "started_at"]].rename(columns={"started_at": "time"}).assign(kind="start")
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
    """A window bound: dates without a time zone are Kyiv calendar dates."""
    x = pd.Timestamp(x)
    return x.tz_localize("Europe/Kyiv") if x.tzinfo is None else x


def monthly_report(labels: pd.DataFrame, reference: pd.DataFrame, start, end) -> pd.DataFrame:
    """Per Kyiv month and event kind: share of reference events found in the labels, and the other way round.
    Months of [start, end) with no events of a kind get a row with `events` 0, `found` NaN and `passes` False."""
    start, end = _kyiv(start), _kyiv(end)
    all_ours, all_theirs = events(labels), events(reference)
    window = lambda ev: ev[(ev.time >= start) & (ev.time < end)].reset_index(drop=True)  # noqa: E731
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
        report = monthly_report(labels, pd.read_parquet(official_path), "2023-01-01", "2026-09-01")
        report.to_csv(out / "official_by_month.csv", index=False)
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
