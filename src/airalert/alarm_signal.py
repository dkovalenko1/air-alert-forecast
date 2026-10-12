"""Read `@UkraineAlarmSignal` (eTryvoga) messages: the alert label source from 2023 (ADR-0006).

Messages are classified by their wording, not their emoji (ADR-0005 rules). Two shapes exist:

    🚨 Шепетівський район (Хмельницька обл.)          one location on the first line,
    Повітряна тривога. Прямуйте в укриття!            the phrase on the second

    🟡 Жовтий рівень тривоги                          the phrase first,
    • Вараський район (Рівненська обл.)               then one location per line
    • Сарненський район (Рівненська обл.)

Only air raid alert starts, all-clears and their continuations/retractions become alert events.
The channel's own city warnings (🛸 drones, 💣 guided bombs, 💥 shelling, ⚠️ media reports…) are
typed as other threats and never open an alert.
"""

import re
from dataclasses import dataclass

START, END, CONTINUE, RETRACT, OTHER, UNKNOWN = "start", "end", "continue", "retract", "other", "unknown"
ESCALATION = "escalation"  # a heightened threat inside an alert (official channel only)


@dataclass(frozen=True)
class Location:
    level: str  # "oblast", "raion" or "city"
    name: str  # as written, e.g. "Шепетівський район", "Харківська обл.", "Київ"
    oblast: str | None  # e.g. "Хмельницька обл."; None when the message does not say


@dataclass(frozen=True)
class Message:
    kind: str
    color: str | None  # "yellow" or "red" for level messages, otherwise None
    threat: str | None  # for OTHER messages: what the warning is about
    locations: tuple[Location, ...]


# (pattern on the phrase, kind, colour). The first match wins, so more specific phrases come first.
_PHRASES = [
    # Apologies retract either an air raid alert or one of the channel's own warnings ("Загроза БПЛА … відсутня");
    # only the first is a retraction. Other apologies (repeated sirens, corrected times) change no alert.
    (r"^просимо вибачення.*(повітряна )?тривога в (області|місті|районі) відсутня", RETRACT, None),
    (r"^просимо вибачення", UNKNOWN, None),
    (r"^відбій тривоги", END, None),
    (r"тривог\w* (в місті |в області )?триває", CONTINUE, None),
    (r"^жовтий рівень тривоги", START, "yellow"),
    (r"^червоний рівень тривоги", START, "red"),
    (r"^повітряна тривога", START, None),
    (r"^тривога\b", START, None),
]
_THREATS = [
    (r"бпла|шахед|дрон", "drones"),
    (r"каб", "guided_bombs"),
    (r"високошвидкісних цілей|ракет|балістик", "missiles"),
    (r"артилерійськ|артобстріл|обстріл", "shelling"),
    (r"вибух", "explosions_reported"),
    (r"вуличн", "urban_fights"),
]

_RAION = re.compile(r"(?P<name>[^()•|]+? район) \((?P<oblast>[^()]+? обл\.)\)")
_CITY = re.compile(r"(?P<name>[^()•|]+?) \((?:[^()]*?, )?(?P<oblast>[^(),]+? обл\.)\)")
_OBLAST = re.compile(r"^(?P<name>[^()•|,]+? обл\.)$")


def _clean(line: str) -> str:
    """Drop leading emoji, bullets and punctuation; normalise apostrophes and spaces."""
    line = re.sub(r"^[^\wА-Яа-яІіЇїЄєҐґ]+", "", line.strip())
    return " ".join(line.replace("’", "'").replace("ʼ", "'").split())


def parse_location(line: str) -> Location | None:
    text = _clean(line)
    if m := _RAION.fullmatch(text):
        return Location("raion", m["name"].strip(), m["oblast"].strip())
    if m := _CITY.fullmatch(text):
        return Location("city", m["name"].strip(), m["oblast"].strip())
    if m := _OBLAST.match(text):
        return Location("oblast", m["name"].strip(), m["name"].strip())
    if text == "Київ":
        return Location("city", "Київ", None)
    return None


def _phrase_kind(phrase: str) -> tuple[str, str | None]:
    lowered = _clean(phrase).lower()
    for pattern, kind, color in _PHRASES:
        if re.search(pattern, lowered):
            return kind, color
    return UNKNOWN, None


def classify(text: str | None) -> Message:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return Message(UNKNOWN, None, None, ())

    first = parse_location(lines[0])
    if first and len(lines) >= 2:  # one location, then the phrase
        locations, phrase = (first,), lines[1]
    else:  # the phrase, then one location per line
        locations = tuple(loc for loc in map(parse_location, lines[1:]) if loc)
        phrase = lines[0]

    kind, color = _phrase_kind(phrase)
    if kind == UNKNOWN and locations and not _clean(phrase).lower().startswith("просимо вибачення"):
        lowered = phrase.lower()
        threat = next((name for pattern, name in _THREATS if re.search(pattern, lowered)), None)
        if threat:
            return Message(OTHER, None, threat, locations)
    if kind in (START, END, CONTINUE, RETRACT) and not locations:
        return Message(UNKNOWN, color, None, ())  # an alert phrase with no recognisable place
    return Message(kind, color, None, locations)
