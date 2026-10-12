"""Read `@air_alert_ua` (the official channel, silent since 2026-09-07): the validation reference (ADR-0006).

Messages look like

    🔴 09:08 Повітряна тривога в Ізюмський район
    Слідкуйте за подальшими повідомленнями.
    #Ізюмський_район

The places are read from the hashtag line, which every alert message has (also when several places are
listed). The kind is read from the wording after the emoji and time, never from the emoji alone
(DESIGN.md, "Message types in @air_alert_ua"):

- "Повітряна тривога в …" starts an alert, "Відбій тривоги в …" ends one (🟡 = oblast all-clear while some
  raion is still on alert; only the oblast-wide alert closes);
- 🔴🔴 "Загроза ударних БпЛА / керованих авіабомб / Ракетна загроза …" are escalations inside an alert;
- artillery, kamikaze-drone, urban-fighting and radiation messages are other threats;
- announcements ("УВАГА!!!", evacuation) are unknown.
"""

import re

from airalert.alarm_signal import END, ESCALATION, OTHER, START, UNKNOWN, Location, Message

_PHRASES = [  # first match wins
    (r"^відбій загроз|^відбій атаки", OTHER),  # end of an escalation or of a non-air threat: changes no alert
    (r"^відбій тривоги", END),
    (r"^повітряна тривога", START),
    (r"^загроза (ударних бпла|керованих авіабомб|застосування каб)|^ракетна загроза", ESCALATION),
    (r"артобстріл|дронів-камікадзе|вуличних боїв|радіаційн|хімічн", OTHER),
]


def _place(tag: str) -> Location:
    name = tag.lstrip("#").replace("_", " ")
    if name.endswith(" район"):
        return Location("raion", name, None)
    if name.endswith(" область"):
        return Location("oblast", name, name)
    if name.startswith("м "):  # "м Київ", "м Харків та Харківська територіальна громада"
        name = "м. " + name[2:]
    return Location("city", name, None)


def classify(text: str | None) -> Message:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return Message(UNKNOWN, None, None, ())
    body = re.sub(r"^[^\wА-Яа-яІіЇїЄєҐґ]*(\d{1,2}:\d{2}\s*)?", "", lines[0]).lower()
    kind = next((k for pattern, k in _PHRASES if re.search(pattern, body)), UNKNOWN)
    tags = re.findall(r"#\S+", lines[-1]) if lines[-1].startswith("#") else []
    locations = tuple(_place(tag) for tag in tags)
    if kind in (START, END, ESCALATION) and not locations:
        return Message(UNKNOWN, None, None, ())
    return Message(kind, None, None, locations)
