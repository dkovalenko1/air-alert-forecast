"""From raw channel messages to alert periods per unit: the source of labels and alert-history features.

Pipeline: messages → classified (alarm_signal / official_channel) → places renamed to one canonical name
(units.canonical, so a start under a raion's old name pairs with an all-clear under its new one) → alert
periods per place (alerts.py) → units (units.py: oblast-wide alerts cover all raions of the oblast, city zones
other than Kyiv are dropped) → periods merged per unit. A unit is on alert whenever any alert covering it is
on, so overlapping periods (an oblast-wide alert plus the raion's own) are merged. Merging never moves a start
later or an end earlier, so asking "was the unit on alert at t, and since when?" stays answerable from data
before t.

Two merged tables come out (ADR-0001): feature periods, as published (a retracted alert lasts until its
retraction, since until then it was believed), and label periods, the corrected truth (retracted alerts dropped,
stretches after a missing all-clear cut out, and the hours before an all-clear for a never-announced alert
unknown). A corrections table records, for every retracted alert and every
unknown stretch, when the correction became known, so training at a cutoff can use labels as known then.

Every message is counted by kind, and unrecognised messages are written out for review (ADR-0005).

Edited messages: Telegram returns only a message's latest text, dated at its original posting time. In the
label channel every message has been edited at least once, mostly in two channel-wide re-edits (2024-01-30 and
2025-06-27) and, since July 2025, automatically about 20 seconds after posting; sampled late edits still carry the
channel's fixed wording. Edits are therefore treated as not changing an alert's kind or place: an evidence-backed
assumption (ADR-0007), cross-checked approximately by the month-by-month comparison with the official channel (Q20).
"""

import dataclasses
import functools
import json
from collections import Counter

import pandas as pd

from airalert import alarm_signal, official_channel
from airalert.alerts import MISSED_START_WINDOW, build_alerts
from airalert.alarm_signal import UNKNOWN
from airalert.settings import RAW_DIR, ROOT
from airalert.units import REGISTRY, UnknownPlace, _oblast_name, canonical, resolve, units

CLASSIFIERS = {"UkraineAlarmSignal": alarm_signal.classify_all, "air_alert_ua": official_channel.classify}
INTERIM_DIR = ROOT / "data" / "interim"
MAX_SILENCE = REGISTRY.parent / "max_silence_hours.csv"  # scripts/calibrate_missing_all_clear.py
MISSING_ALL_CLEAR_RULE = {"UkraineAlarmSignal"}  # the official channel is the reference and is left as published


@functools.cache
def _silence_limits() -> dict[str, pd.Timedelta]:
    table = pd.read_csv(MAX_SILENCE)
    return {row.oblast: pd.Timedelta(hours=row.max_silence_hours) for row in table.itertuples()}


def max_silence(location) -> pd.Timedelta | None:
    """Longest plausible stretch without news about an open alert at `location` (its oblast's limit)."""
    if location.level == "city" and location.oblast is None:  # Kyiv
        return _silence_limits().get("м. Київ")
    return _silence_limits().get(_oblast_name(location.oblast)) if location.oblast else None


def read_channel(channel: str) -> list[dict]:
    messages = []
    for path in sorted((RAW_DIR / "telegram" / channel).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            messages.extend(json.loads(line) for line in f)
    return sorted(messages, key=lambda m: m["id"])


def classify_messages(messages: list[dict], classify) -> tuple[list, Counter, list[dict]]:
    """(id, time, message) with places renamed to their canonical names; counts per kind; unrecognised messages.
    `classify` returns one Message or a list (one per section; see alarm_signal.classify_all). A place not in the
    registry is counted and dropped from its message."""
    parsed, stats, unknown = [], Counter(), []
    for m in messages:
        if not m.get("text"):  # service messages (channel created, pinned…) and media without a caption
            stats["message: service" if m.get("service") else "message: no text"] += 1
            continue
        sections = classify(m["text"])
        sections = sections if isinstance(sections, list) else [sections]
        if len(sections) > 1:
            stats["message with several sections"] += 1
        if all(section.kind == UNKNOWN for section in sections):
            unknown.append({"id": m["id"], "date": m["date"], "text": m["text"]})
        for message in sections:
            stats[f"message: {message.kind}"] += 1
            locations = []
            for location in message.locations:
                try:
                    locations.append(canonical(location))
                except UnknownPlace as error:
                    stats[f"unknown place: {error}"] += 1
            parsed.append((m["id"], pd.Timestamp(m["date"]), dataclasses.replace(message, locations=tuple(locations))))
    return parsed, stats, unknown


def unit_alerts_from_messages(messages: list[dict], classify, max_silence=None) -> tuple[pd.DataFrame, Counter, list[dict]]:
    """One row per (alert, unit it covers). `finished_at` is NaT for alerts still open when the data ends;
    `retracted` marks alerts ended by a retraction; `unknown` lists (from, to, detected_at) stretches after a missing
    all-clear (only with `max_silence`, see alerts.py). Also returns counts and the unrecognised messages."""
    parsed, stats, unknown = classify_messages(messages, classify)
    data_end = pd.Timestamp(messages[-1]["date"]) if messages else None
    history = (build_alerts(parsed, max_silence=max_silence, data_end=data_end, missed_start_window=MISSED_START_WINDOW)
               if max_silence else build_alerts(parsed))
    stats.update(history.stats)
    excluded = {u.unit_id for u in units() if u.excluded}
    rows = []
    for alert in history.alerts:
        try:
            unit_ids = resolve(alert.location)
        except UnknownPlace as error:
            stats[f"unknown place: {error}"] += 1
            continue
        for unit_id in unit_ids:
            if unit_id in excluded:
                continue
            rows.append({
                "unit_id": unit_id,
                "started_at": alert.started_at,
                "finished_at": alert.finished_at,
                "place_level": alert.location.level,
                "place": alert.location.name,
                "max_color": alert.max_color,
                "implied": alert.implied,
                "restarts": alert.restarts,
                "retracted": alert.retracted,
                "missed_start": alert.missed_start,
                "unknown": list(alert.unknown),
            })
    columns = ["unit_id", "started_at", "finished_at", "place_level", "place", "max_color", "implied", "restarts", "retracted", "missed_start", "unknown"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("started_at", "finished_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["retracted"] = frame["retracted"].astype(bool)
    frame["missed_start"] = frame["missed_start"].astype(bool)
    return frame, stats, unknown


def unknown_periods(alerts: pd.DataFrame) -> pd.DataFrame:
    """Stretches with unknown state after a missing all-clear, per unit: unit_id, started_at, finished_at, corrected_at
    (when it became clear the all-clear was missing). Retracted alerts are dropped from labels anyway."""
    rows = [(row.unit_id, a, b, d) for row in alerts[~alerts.retracted].itertuples() for a, b, d in (row.unknown or [])]
    frame = pd.DataFrame(rows, columns=["unit_id", "started_at", "finished_at", "corrected_at"])
    for column in ("started_at", "finished_at", "corrected_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def known_pieces(alerts: pd.DataFrame) -> pd.DataFrame:
    """The non-retracted alerts with their unknown stretches cut out: what labels count as on alert."""
    rows = []
    for row in alerts[~alerts.retracted].itertuples():
        start = row.started_at
        for a, b, _ in sorted(row.unknown or []):
            if a > start:
                rows.append((row.unit_id, start, pd.Timestamp(a)))
            start = max(start, pd.Timestamp(b))
        end = row.finished_at
        if pd.isna(end) or end > start:
            rows.append((row.unit_id, start, end))
    frame = pd.DataFrame(rows, columns=["unit_id", "started_at", "finished_at"])
    for column in ("started_at", "finished_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def corrections(alerts: pd.DataFrame) -> pd.DataFrame:
    """Every label correction and when it became known: retracted alerts (label → calm, known at the retraction)
    and unknown stretches (label → unknown, known when the missing all-clear could be detected)."""
    retracted = alerts.loc[alerts.retracted, ["unit_id", "started_at", "finished_at"]]
    retracted = retracted.assign(corrected_at=retracted.finished_at, unknown=False)
    return pd.concat([retracted, unknown_periods(alerts).assign(unknown=True)], ignore_index=True)


def merge_per_unit(alerts: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """Union of alert periods per unit. Still-open alerts run to `data_end` and are flagged."""
    open_end = alerts["finished_at"].isna()
    alerts = alerts.assign(finished_at=alerts["finished_at"].fillna(data_end), open_at_end=open_end)
    merged = []
    for unit_id, group in alerts.sort_values("started_at").groupby("unit_id", sort=True):
        start = end = None
        is_open = False
        for row in group.itertuples(index=False):
            if start is not None and row.started_at <= end:
                end, is_open = max(end, row.finished_at), is_open or row.open_at_end
                continue
            if start is not None:
                merged.append((unit_id, start, end, is_open))
            start, end, is_open = row.started_at, row.finished_at, row.open_at_end
        if start is not None:
            merged.append((unit_id, start, end, is_open))
    return pd.DataFrame(merged, columns=["unit_id", "started_at", "finished_at", "open_at_end"])


def build(channel: str) -> None:
    """Write data/interim/<channel>_alerts.parquet (per alert and unit), <channel>_merged.parquet (feature periods),
    <channel>_label_periods.parquet (label periods), <channel>_corrections.parquet (retracted alerts and unknown
    stretches, with when each became known) and <channel>_unknown.csv (unrecognised messages)."""
    messages = read_channel(channel)
    if not messages:
        raise SystemExit(f"No downloaded messages for {channel}: run `download-telegram {channel}` first.")
    rule = max_silence if channel in MISSING_ALL_CLEAR_RULE else None
    alerts, stats, unknown = unit_alerts_from_messages(messages, CLASSIFIERS[channel], max_silence=rule)
    data_end = pd.Timestamp(messages[-1]["date"])
    merged = merge_per_unit(alerts[~alerts.missed_start], data_end)  # as published: a never-announced alert is not in it
    label_periods = merge_per_unit(known_pieces(alerts), data_end)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    alerts.drop(columns="unknown").to_parquet(INTERIM_DIR / f"{channel}_alerts.parquet", index=False)
    merged.assign(data_end=data_end).to_parquet(INTERIM_DIR / f"{channel}_merged.parquet", index=False)
    label_periods.assign(data_end=data_end).to_parquet(INTERIM_DIR / f"{channel}_label_periods.parquet", index=False)
    corrections(alerts).to_parquet(INTERIM_DIR / f"{channel}_corrections.parquet", index=False)
    pd.DataFrame(unknown, columns=["id", "date", "text"]).to_csv(INTERIM_DIR / f"{channel}_unknown.csv", index=False)
    print(f"{channel}: {len(messages):,} messages up to {data_end:%Y-%m-%d %H:%M} UTC → "
          f"{len(alerts):,} unit alerts, {len(merged):,} merged periods for {merged.unit_id.nunique()} units; "
          f"{len(unknown):,} unrecognised messages in {channel}_unknown.csv")
    for name, count in stats.most_common():
        print(f"  {count:6,}  {name}")
