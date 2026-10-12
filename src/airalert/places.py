"""Find places in Telegram threat posts and map them to units (Q12). Rule-based, no language model.

Recognised, in any grammatical case:
- cities and settlements from reference/places.csv ("на Павлоград", "в районі Чорткова", "Білої Церкви");
- raions ("Ізюмського району");
- oblasts, by region name ("з Полтавщини на Черкащину") or by name ("Харківської області"). An oblast
  mention covers all raions of that oblast.

Ukrainian inflection is handled by stems plus case endings, with the і→о alternation of names like
Харків → Харкова, Київ → Києва. A few irregular names have their forms listed. Month names are never
matched (the town Березне would otherwise match "березня", March).
"""

import csv
import functools
import pathlib
import re
from dataclasses import dataclass

from airalert.alarm_signal import Location
from airalert.units import resolve, units

PLACES = pathlib.Path(__file__).resolve().parent / "reference" / "places.csv"
ENDINGS = sorted({"", "а", "я", "у", "ю", "і", "ї", "е", "є", "о", "ом", "ем", "єм", "ою", "ею", "єю", "ові", "еві", "єві",
                  "и", "ах", "ях", "ам", "ям", "ами", "ями", "ів", "ий", "ого", "ому", "ій", "ої", "ім", "им", "ими", "их"},
                 key=len, reverse=True)
NEVER = {"січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада",
         "грудня", "щастя", "сума", "суму", "сумою"}
IRREGULAR = {"кривий ріг": ["кривий ріг", "кривого рогу", "кривому розі", "кривим рогом", "кр"]}  # "кр" is never alone
# Region names → oblast. Most follow "<stem>щин/ччин + case ending"; the rest are listed with their forms.
REGION_STEMS = {
    "вінниччин": "Вінницька", "дніпропетровщин": "Дніпропетровська", "дніпровщин": "Дніпропетровська",
    "донеччин": "Донецька", "житомирщин": "Житомирська", "запоріжчин": "Запорізька", "франківщин": "Івано-Франківська",
    "київщин": "Київська", "кіровоградщин": "Кіровоградська", "кропивниччин": "Кіровоградська",
    "луганщин": "Луганська", "львівщин": "Львівська", "миколаївщин": "Миколаївська", "одещин": "Одеська",
    "полтавщин": "Полтавська", "рівненщин": "Рівненська", "сумщин": "Сумська", "тернопільщин": "Тернопільська",
    "харківщин": "Харківська", "херсонщин": "Херсонська", "хмельниччин": "Хмельницька", "черкащин": "Черкаська",
    "чернівеччин": "Чернівецька", "буковин": "Чернівецька", "чернігівщин": "Чернігівська",
}
REGION_FORMS = {
    "Волинська": ["волинь", "волині", "волинню"],
    "Закарпатська": ["закарпаття", "закарпатті", "закарпаттям"],
    "Івано-Франківська": ["прикарпаття", "прикарпатті", "прикарпаттям"],
}
REGION_ENDINGS = ("а", "и", "у", "і", "ою")
TOKEN = re.compile(r"[а-щьюяєіїґ'’ʼ-]+", re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    kind: str  # "place", "raion" or "oblast"
    name: str
    unit_ids: tuple[str, ...]


def _lower(text: str) -> str:
    return text.lower().replace("’", "'").replace("ʼ", "'")


def _stems(word: str) -> set[str]:
    stems = {word}
    if word[-1] in "аяоеєіиьй":
        stems.add(word[:-1])
    if word.endswith(("ий", "ій")):  # adjectives: Ізюмський → Ізюмського, Ізюмському
        stems.add(word[:-2])
    if word.endswith("ів"):
        stems.add(word[:-2] + "ов")
    if word.endswith("їв"):
        stems.add(word[:-2] + "єв")
    return {s for s in stems if len(s) >= 3}


@functools.cache
def _index():
    """stem tuple (one stem per word of the name) → Mention."""
    index, phrases = {}, {}
    with PLACES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _lower(row["name"])
            mention = Mention("place", row["name"], (row["unit_id"],))
            if name in IRREGULAR:
                for form in IRREGULAR[name]:
                    phrases[tuple(form.split())] = mention
                continue
            words = name.split()
            stem_options = [_stems(w) for w in words]
            for combo in _combinations(stem_options):
                index[combo] = mention
    for unit in units():
        if unit.kind != "raion":
            continue
        adjective = _lower(unit.name.removesuffix(" район"))
        for stem in _stems(adjective):
            index[(stem, "район")] = Mention("raion", unit.name, (unit.unit_id,))
    for oblast in {u.oblast for u in units() if u.kind == "raion"}:
        covered = tuple(resolve(Location("oblast", oblast, oblast)))
        adjective = _lower(oblast.removesuffix(" область"))
        for stem in _stems(adjective):
            index[(stem, "област")] = Mention("oblast", oblast, covered)
    for stem, oblast in REGION_STEMS.items():
        name = f"{oblast} область"
        covered = tuple(resolve(Location("oblast", name, name)))
        for ending in REGION_ENDINGS:
            phrases[(stem + ending,)] = Mention("oblast", name, covered)
    for oblast, forms in REGION_FORMS.items():
        name = f"{oblast} область"
        covered = tuple(resolve(Location("oblast", name, name)))
        for form in forms:
            phrases[(form,)] = Mention("oblast", name, covered)
    return index, phrases


def _combinations(options):
    if not options:
        yield ()
        return
    for first in options[0]:
        for rest in _combinations(options[1:]):
            yield (first, *rest)


def _candidate_stems(token: str) -> list[str]:
    return [token[: len(token) - len(e)] for e in ENDINGS if token.endswith(e) and len(token) - len(e) >= 3]


def find_places(text: str | None) -> list[Mention]:
    """Places mentioned in `text`, in order, each once."""
    index, phrases = _index()
    tokens = [t.strip("-'") for t in TOKEN.findall(_lower(text or ""))]
    found, i = [], 0
    while i < len(tokens):
        match, width = None, 1
        for n in (3, 2, 1):  # longest phrase first
            phrase = tuple(tokens[i:i + n])
            if len(phrase) == n and phrase in phrases:
                match, width = phrases[phrase], n
                break
        if match is None and tokens[i] not in NEVER:
            for n in (2, 1):
                window = tokens[i:i + n]
                if len(window) < n:
                    continue
                # "<adjective> район/району/районі" or "<adjective> області/область"
                if n == 2 and window[1].startswith(("район", "област")):
                    tail = "район" if window[1].startswith("район") else "област"
                    match = next((index[(s, tail)] for s in _candidate_stems(window[0]) if (s, tail) in index), None)
                else:
                    stem_lists = [_candidate_stems(w) for w in window]
                    match = next((index[c] for c in _combinations(stem_lists) if c in index), None)
                if match:
                    width = n
                    break
        if match and match not in found:
            found.append(match)
        i += width
    return found
