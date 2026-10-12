"""Turn classified alert messages into alert periods, following the Q17 rules in docs/DESIGN.md.

Each location is tracked on its own; mapping oblast-wide alerts onto raions happens later, when units are built.

- A start opens an alert. A repeated start for a location already on alert keeps the original start
  time (a yellow↔red change continues the same alert, Q18) and is counted as a restart.
- An all-clear closes the open alert. An all-clear with nothing open is counted and ignored.
- A retraction ("Просимо вибачення, сталась технічна помилка…") removes the open alert: it was posted by mistake.
- A continuation ("тривога триває") changes nothing.
- An escalation ("Загроза ударних БпЛА…", official channel) never changes an open alert; with nothing open
  it opens an alert flagged as implied.
- Alerts still open when the data ends keep `finished_at = None`.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

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


@dataclass
class AlertHistory:
    alerts: list[Alert] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def _raise_color(current: str | None, new: str | None) -> str | None:
    return "red" if "red" in (current, new) else new or current


def build_alerts(messages: list[tuple[int, datetime, Message]]) -> AlertHistory:
    """`messages` are (message_id, time, classified message) in channel order."""
    history, open_alerts = AlertHistory(), {}
    for message_id, time, message in messages:
        for location in message.locations:
            current = open_alerts.get(location)
            if message.kind == START:
                if current:
                    current.restarts += 1
                    current.max_color = _raise_color(current.max_color, message.color)
                    history.stats["restart while on alert"] += 1
                else:
                    open_alerts[location] = Alert(location, time, message_id, max_color=message.color)
            elif message.kind == END:
                if current:
                    current.finished_at, current.end_message_id = time, message_id
                    history.alerts.append(open_alerts.pop(location))
                else:
                    history.stats["all-clear with no open alert"] += 1
            elif message.kind == RETRACT:
                history.stats["retracted" if open_alerts.pop(location, None) else "retraction with no open alert"] += 1
            elif message.kind == ESCALATION:
                if current:
                    history.stats["escalation during alert"] += 1
                else:
                    open_alerts[location] = Alert(location, time, message_id, implied=True)
                    history.stats["implied alert from escalation"] += 1
            elif message.kind == CONTINUE and not current:
                history.stats["continuation with no open alert"] += 1
    history.alerts.extend(open_alerts.values())  # still on alert when the data ends
    history.alerts.sort(key=lambda a: (a.started_at, a.start_message_id))
    return history
