"""Turn classified alert messages into alert periods, following the Q17 rules in docs/DESIGN.md.

Each location is tracked on its own; mapping oblast-wide alerts onto raions happens later, when units are built.

- A start opens an alert. A repeated start for a location already on alert keeps the original start
  time (a yellow↔red change continues the same alert, Q18) and is counted as a restart.
- An all-clear closes the open alert. An all-clear with nothing open is counted and ignored.
- A retraction ("Просимо вибачення, сталась технічна помилка…") closes the open alert at the retraction and flags it
  `retracted`. Until the retraction was posted, everyone believed the alert was on, so features (what was known at
  t) keep it up to that moment; labels (what really happened) drop it.
- A continuation ("тривога триває") changes nothing.
- An escalation ("Загроза ударних БпЛА…", official channel) never changes an open alert; with nothing open
  it opens an alert flagged as implied.
- Alerts still open when the data ends keep `finished_at = None`.
- A missing all-clear (Q17) leaves the alert as published (features use it: nobody knew the all-clear was missing),
  but marks the stretch with no information as unknown, so labels neither train nor score on it. A stretch
  between two messages about the place is unknown if
  - a new start without a level arrives more than `restart_gap` (20 min) after the previous message: the
    all-clear in between was lost (in the official channel the place was calm in between in 86-100% of such
    cases); detected when that start arrives;
  - or it is longer than the place's `max_silence` (per oblast, calibrated on the official channel by
    scripts/calibrate_missing_all_clear.py); detected once the silence passes that limit.
  Level starts (yellow ↔ red) are level changes, not lost all-clears, so only the silence rule applies to them.
- A missing start: an all-clear for a place with no open alert, more than `restart_gap` after the previous
  all-clear there (closer ones are repeated posts). An alert was on but never announced (the official channel was
  on alert just before such all-clears in 90-98% of cases). With `missed_start_window`, it becomes an alert flagged
  `missed_start` whose whole span is unknown: from the window (5 h: 95% of those official alerts started within it)
  or the place's previous message, whichever is later, to the all-clear; detected at the all-clear. It was never
  published, so features ignore it.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from airalert.alarm_signal import CONTINUE, END, ESCALATION, RETRACT, START, Location, Message


@dataclass
class Alert:
    location: Location
    started_at: datetime
    start_message_id: int
    finished_at: datetime | None = None
    end_message_id: int | None = None
    max_color: str | None = None  # "red" if the alert was ever red, else "yellow" if ever yellow
    restarts: int = 0
    implied: bool = False  # opened by an escalation with no air raid alert open
    retracted: bool = False  # posted by mistake: ended by a retraction, not an all-clear
    missed_start: bool = False  # never announced: known only from an all-clear with no open alert
    last_message_at: datetime | None = None  # the latest message about the place while the alert was open
    unknown: list = field(default_factory=list)  # (from, to, detected_at): stretches whose state is unknown


@dataclass
class AlertHistory:
    alerts: list[Alert] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def _raise_color(current: str | None, new: str | None) -> str | None:
    return "red" if "red" in (current, new) else new or current


RESTART_GAP = timedelta(minutes=20)
MISSED_START_WINDOW = timedelta(hours=5)


def _check_silence(alert: Alert, until: datetime, max_silence, stats: Counter) -> None:
    """Mark the stretch from the alert's last message to `until` unknown if it is longer than the place's limit."""
    limit = max_silence(alert.location) if max_silence else None
    if limit is not None and until - alert.last_message_at > limit:
        alert.unknown.append((alert.last_message_at, until, alert.last_message_at + limit))
        stats["missing all-clear: long silence"] += 1


def build_alerts(messages: list[tuple[int, datetime, Message]], max_silence=None, restart_gap=RESTART_GAP,
                 data_end: datetime | None = None, missed_start_window: timedelta | None = None) -> AlertHistory:
    """`messages` are (message_id, time, classified message) in channel order. `max_silence(location)` gives the
    longest plausible stretch without news about an open alert (None: no limit); `data_end` is when the data ends;
    `missed_start_window` turns all-clears with no open alert into unknown stretches (None: only counted)."""
    history, open_alerts, last_seen = AlertHistory(), {}, {}
    for message_id, time, message in messages:
        for location in message.locations:
            current = open_alerts.get(location)
            previous = last_seen.get(location)  # (time, kind) of the previous message about the place
            last_seen[location] = (time, message.kind)
            if current and message.kind != RETRACT:
                silence_before = len(current.unknown)
                _check_silence(current, time, max_silence, history.stats)
                if (message.kind == START and message.color is None and len(current.unknown) == silence_before
                        and time - current.last_message_at > restart_gap):
                    current.unknown.append((current.last_message_at, time, time))
                    history.stats["missing all-clear: restart"] += 1
                if message.kind in (START, CONTINUE, ESCALATION):
                    current.last_message_at = time
            if message.kind == START:
                if current:
                    current.restarts += 1
                    current.max_color = _raise_color(current.max_color, message.color)
                    history.stats["restart while on alert"] += 1
                else:
                    open_alerts[location] = Alert(location, time, message_id, max_color=message.color, last_message_at=time)
            elif message.kind == END:
                if current:
                    current.finished_at, current.end_message_id = time, message_id
                    history.alerts.append(open_alerts.pop(location))
                else:
                    history.stats["all-clear with no open alert"] += 1
                    repeated = previous is not None and previous[1] == END and time - previous[0] <= restart_gap
                    if missed_start_window is not None and not repeated:
                        start = max(time - missed_start_window, previous[0]) if previous else time - missed_start_window
                        history.alerts.append(Alert(location, start, message_id, finished_at=time, end_message_id=message_id,
                                                    missed_start=True, unknown=[(start, time, time)]))
                        history.stats["missing start: unknown stretch before an all-clear"] += 1
            elif message.kind == RETRACT:
                if current:
                    current.finished_at, current.end_message_id, current.retracted = time, message_id, True
                    history.alerts.append(open_alerts.pop(location))
                    history.stats["retracted"] += 1
                else:
                    history.stats["retraction with no open alert"] += 1
            elif message.kind == ESCALATION:
                if current:
                    history.stats["escalation during alert"] += 1
                else:
                    open_alerts[location] = Alert(location, time, message_id, implied=True, last_message_at=time)
                    history.stats["implied alert from escalation"] += 1
            elif message.kind == CONTINUE and not current:
                history.stats["continuation with no open alert"] += 1
    for alert in open_alerts.values():  # still on alert when the data ends
        if data_end is not None:
            _check_silence(alert, data_end, max_silence, history.stats)
        history.alerts.append(alert)
    history.alerts.sort(key=lambda a: (a.started_at, a.start_message_id))
    return history
