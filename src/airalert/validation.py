"""Label-quality checks: our label source against independent records (Q20, Q2, ADR-0006).

Alert starts and all-clears are compared per unit: an event in one source counts as found if the other source
has the same kind of event for the same unit within 5 minutes. Both directions are reported per month, against
the pass mark fixed in advance (Q20): at least 98% each way, every month.

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
    """Start and all-clear events per unit, one row per distinct (unit, kind, time)."""
    starts = alerts[["unit_id", "started_at"]].rename(columns={"started_at": "time"}).assign(kind="start")
    ends = alerts.dropna(subset=["finished_at"])[["unit_id", "finished_at"]].rename(columns={"finished_at": "time"}).assign(kind="end")
    return pd.concat([starts, ends]).drop_duplicates().sort_values("time", ignore_index=True)


def found_share(source: pd.DataFrame, other: pd.DataFrame, tolerance=TOLERANCE) -> pd.Series:
    """For each event in `source`: is there an event of the same kind and unit in `other` within `tolerance`?"""
    found = np.zeros(len(source), dtype=bool)
    for (unit_id, kind), group in source.groupby(["unit_id", "kind"]):
        times = np.sort(other.loc[(other.unit_id == unit_id) & (other.kind == kind), "time"].to_numpy())
        if not len(times):
            continue
        x = group.time.to_numpy()
        i = np.clip(np.searchsorted(times, x), 1, len(times)) - 1
        nearest = np.minimum(np.abs(times[i] - x), np.abs(times[np.minimum(i + 1, len(times) - 1)] - x))
        found[source.index.get_indexer(group.index)] = nearest <= tolerance.to_timedelta64()
    return pd.Series(found, index=source.index)


def monthly_report(labels: pd.DataFrame, reference: pd.DataFrame, start, end) -> pd.DataFrame:
    """Per month and event kind: share of reference events found in the labels, and the other way round."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    window = lambda ev: ev[(ev.time >= start) & (ev.time < end)].reset_index(drop=True)  # noqa: E731
    ours, theirs = window(events(labels)), window(events(reference))
    theirs["found"] = found_share(theirs, ours).to_numpy()
    ours["found"] = found_share(ours, theirs).to_numpy()
    rows = []
    for direction, ev in (("reference found in labels", theirs), ("labels found in reference", ours)):
        month = ev.time.dt.tz_convert("Europe/Kyiv").dt.strftime("%Y-%m")
        for (m, kind), group in ev.groupby([month, ev.kind]):
            rows.append({"month": m, "kind": kind, "direction": direction, "events": len(group), "found": group.found.mean()})
    report = pd.DataFrame(rows)
    report["passes"] = report.found >= PASS_MARK
    return report


def alerts_in_ua_alerts() -> pd.DataFrame:
    """alerts.in.ua air raid alerts as unit alerts (same shape as labels.unit_alerts_from_messages).

    Raions map to themselves and oblasts to all their raions; Kyiv city to its unit. Other city and hromada
    records are dropped, as in our labels (Q16). Snapshots overlap, so alerts are de-duplicated by id.
    """
    excluded = {u.unit_id for u in units() if u.excluded}
    seen, rows = set(), []
    for path in sorted(glob.glob(str(RAW_DIR / "alerts_in_ua" / "*" / "*.json"))):
        for alert in json.load(open(path, encoding="utf-8"))["alerts"]:
            if alert["id"] in seen or alert["alert_type"] != "air_raid":
                continue
            seen.add(alert["id"])
            level = {"raion": "raion", "oblast": "oblast", "city": "city"}.get(alert["location_type"])
            if level is None:
                continue
            oblast = alert.get("location_oblast")
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
    report = monthly_report(labels, alerts_in_ua_alerts(), start.tz_convert("UTC").tz_localize(None), end.tz_convert("UTC").tz_localize(None))
    report.to_csv(out / "alerts_in_ua_test_period.csv", index=False)
    print("Q2 (test period, vs alerts.in.ua):")
    print(report.to_string(index=False))
