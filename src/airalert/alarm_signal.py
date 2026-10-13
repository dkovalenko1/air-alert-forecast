"""Read `@UkraineAlarmSignal` (eTryvoga) messages: the alert label source from 2023 (ADR-0006).

Messages are classified by their wording, not their emoji (ADR-0005 rules). Two shapes exist:

    🚨 Шепетівський район (Хмельницька обл.)          one location on the first line,
    Повітряна тривога. Прямуйте в укриття!            the phrase on the second

    🟡 Жовтий рівень тривоги                          the phrase first,
    • Вараський район (Рівненська обл.)               then one location per line
    • Сарненський район (Рівненська обл.)

Since July 2026 one message may hold several sections, separated by a blank line, each with its own heading:

    🟡 Відбій червоної тривоги                        red is over, yellow continues (a yellow start)
    • Чугуївський район (Харківська обл.)

    🟢 Відбій тривоги                                 all-clear
    • Ізюмський район (Харківська обл.)

`classify_all` reads each section separately; `classify` reads a single section.

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


# Apologies are read as a whole message body (the key sentence may be on a later line). They retract either an
# air raid alert or one of the channel's own warnings ("Загроза БПЛА … відсутня"); only the first is a retraction.
# An apology giving a late all-clear ("…відбій повітряної тривоги був о 22:29") ends the alert when it is posted:
# the earlier time it names was not known before the message. Other apologies change no alert.
_APOLOGY = re.compile(r"^(просимо|приносимо|перепрошуємо|вибач)|^\w+ за помилков")
_APOLOGY_KINDS = [
    (r"тривог\w*( у| в| по)?[\w' ]*? (області|місті|районі|громаді) (відсутня|не було|немає)"
     r"|тривог\w*( в області| у місті| в місті)? (не було|немає)|не було (повітряної )?тривоги"
     r"|тривог\w* (була |було )?помилков|за помилкову тривогу", RETRACT),
    (r"відбій (повітряної )?тривоги\W*(був|оголошен|в |у |о |з )|відбій (у|в) [\w' ]+ районі був", END),
]
# (pattern on the phrase, kind, colour). The first match wins, so more specific phrases come first.
_PHRASES = [
    (r"^(повітряна )?тривога\b.*\bпомилков", RETRACT, None),  # "Тривога в області була помилковою…"
    (r"^відбій тривоги", END, None),
    # red → yellow: a yellow alert is on. Read as a yellow start: inside an open alert it is a level change; with none
    # open (the channel missed the start) it opens one, as alerts.in.ua shows for such cases.
    (r"^відбій червоної тривоги", START, "yellow"),
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


def _apology_kind(body: str) -> str:
    return next((kind for pattern, kind in _APOLOGY_KINDS if re.search(pattern, body)), UNKNOWN)


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

    apology = bool(_APOLOGY.search(_clean(phrase).lower()))
    if apology:  # the whole body after the place line(s)
        body = " ".join(_clean(line) for line in (lines[1:] if first and len(lines) >= 2 else lines)).lower()
        kind, color = _apology_kind(body), None
    else:
        kind, color = _phrase_kind(phrase)
    if kind == UNKNOWN and locations and not apology:
        lowered = phrase.lower()
        threat = next((name for pattern, name in _THREATS if re.search(pattern, lowered)), None)
        if threat:
            return Message(OTHER, None, threat, locations)
    if kind in (START, END, CONTINUE, RETRACT) and not locations:
        return Message(UNKNOWN, color, None, ())  # an alert phrase with no recognisable place
    return Message(kind, color, None, locations)


def classify_all(text: str | None) -> list[Message]:
    """Every section of a message, in order. A message is split at blank lines only if at least two of its parts
    start with an alert heading; otherwise it is one section (blank lines also occur inside ordinary posts)."""
    blocks = [b for b in re.split(r"\n\s*\n", (text or "").strip()) if b.strip()]
    headed = [b for b in blocks if _phrase_kind(b.strip().splitlines()[0])[0] != UNKNOWN]
    if len(headed) < 2:
        return [classify(text)]
    return [classify(b) for b in blocks]
