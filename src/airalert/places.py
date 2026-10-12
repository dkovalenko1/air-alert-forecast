"""Find places in Telegram threat posts and map them to units (Q12). Rule-based, no language model.

Recognised, in any grammatical case:
- cities and settlements from reference/places.csv ("на Павлоград", "в районі Чорткова", "Білої Церкви");
- raions ("Ізюмського району");
- oblasts, by region name ("з Полтавщини на Черкащину") or by name ("Харківської області"). An oblast
  mention covers all raions of that oblast.

Ukrainian inflection is handled by stems plus the case endings each word's type can take: a feminine name
in -а (Лозова → Лозову, Лозової), a neuter adjective in -е (Покровське → Покровського), a masculine name
(Покровськ → Покровська) and so on, with the і→о alternation of names like Харків → Харкова, Київ → Києва.
So "на Лозову" can only be Лозова, and "на Покровськ" only Покровськ. If places of the same word type still
share a stem, a city wins over a settlement; places that remain tied in different raions are not matched,
rather than guessed. A few irregular names have their forms listed. Month names are never matched (the town
Березне would otherwise match "березня", March). Raions are also found by their former names (registry aliases).
"""

import csv
import functools
import pathlib
import re
from dataclasses import dataclass

from airalert.alarm_signal import Location
from airalert.units import resolve, units

PLACES = pathlib.Path(__file__).resolve().parent / "reference" / "places.csv"
_ENDINGS = {"", "а", "я", "у", "ю", "і", "ї", "е", "є", "о", "ом", "ем", "єм", "ою", "ею", "єю", "ові", "еві", "єві",
            "и", "ах", "ях", "ам", "ям", "ами", "ями", "ів", "ий", "ого", "ому", "ій", "ої", "ім", "им", "ими", "их"}
NEVER = {"січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада",
         "грудня", "щастя", "сума", "суму", "сумою"}
# "КР" is not listed for Кривий Ріг: in Air Force posts it usually means cruise missiles (крилаті ракети).
IRREGULAR = {"кривий ріг": ["кривий ріг", "кривого рогу", "кривому розі", "кривим рогом"],
             "часів яр": ["часів яр", "часового яру", "часовому яру", "часовим яром", "часовому ярі"]}
# Case endings by word type, chosen by how the name ends in the nominative (see _word_forms).
WORD_ENDINGS = {
    "masc_adj": {"ий", "ій", "ого", "ому", "им", "ім", "ього", "ьому"},  # Великий, Верхній
    "fem": {"а", "и", "і", "у", "ою", "ої", "ій", "о"},  # Лозова, Буча, Біла Церква
    "fem_soft": {"я", "і", "ї", "ю", "ею", "єю", "ій", "ям"},  # Знам'янка-like soft names, Запоріжжя
    "neut_adj": {"е", "є", "ого", "ому", "им", "ім", "ього", "ьому"},  # Покровське, Верхнє
    "neut": {"о", "а", "у", "і", "ом", "ові"},  # Дніпро
    "plural": {"", "і", "и", "ів", "ам", "ям", "ах", "ях", "ами", "ями", "ей", "их", "им", "ими"},  # Суми, Великі Мости
    "soft": {"ь", "я", "ю", "ем", "і", "еві", "й", "є", "єм", "ї"},  # Ірпінь
    "masc": {"", "а", "у", "ові", "еві", "ом", "ем", "і"},  # Покровськ, Харків
}
PRIORITY = {"city": 0, "settlement": 1}
ENDINGS = sorted(_ENDINGS.union(*WORD_ENDINGS.values()), key=len, reverse=True)
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


def _word_forms(word: str) -> list[tuple[str, frozenset, str]]:
    """(stem, allowed endings, the nominative's own ending) for one word of a place name, by how it ends."""
    if word.endswith(("ий", "ій")):
        kind, stem = "masc_adj", word[:-2]
    elif word[-1] in "ая":
        kind, stem = ("fem" if word[-1] == "а" else "fem_soft"), word[:-1]
    elif word[-1] in "еє":
        kind, stem = "neut_adj", word[:-1]
    elif word[-1] == "о":
        kind, stem = "neut", word[:-1]
    elif word[-1] in "иі":
        kind, stem = "plural", word[:-1]
    elif word[-1] in "ьй":
        kind, stem = "soft", word[:-1]
    else:
        kind, stem = "masc", word
    endings = frozenset(WORD_ENDINGS[kind] | {word[len(stem):]})
    forms = [(stem, endings, word[len(stem):])]
    if word.endswith("ів"):  # Харків → Харкова
        forms.append((word[:-2] + "ов", endings, None))
    if word.endswith("їв"):  # Київ → Києва
        forms.append((word[:-2] + "єв", endings, None))
    return [form for form in forms if len(form[0]) >= 3]


@functools.cache
def _index():
    """Places: stem tuple (one stem per word) → [(Mention, allowed endings per word, nominative endings, priority)].
    Raions and oblasts: (adjective stem, "район" / "област") → Mention."""
    places, index, phrases = {}, {}, {}
    with PLACES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _lower(row["name"])
            mention = Mention("place", row["name"], (row["unit_id"],))
            if name in IRREGULAR:
                for form in IRREGULAR[name]:
                    phrases[tuple(form.split())] = mention
                continue
            for combo in _combinations([_word_forms(w) for w in name.split()]):
                stems = tuple(form[0] for form in combo)
                allowed, nominative = tuple(form[1] for form in combo), tuple(form[2] for form in combo)
                places.setdefault(stems, []).append((mention, allowed, nominative, PRIORITY[row["kind"]]))
    for unit in units():
        if unit.kind != "raion":
            continue
        for name in [unit.name, *unit.aliases]:  # former names too (Новомосковський → Самарівський)
            adjective = _lower(name.removesuffix(" район"))
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
    return places, index, phrases


def _combinations(options):
    if not options:
        yield ()
        return
    for first in options[0]:
        for rest in _combinations(options[1:]):
            yield (first, *rest)


def _candidate_stems(token: str) -> list[str]:
    return [token[: len(token) - len(e)] for e in ENDINGS if token.endswith(e) and len(token) - len(e) >= 3]


def _splits(token: str) -> list[tuple[str, str]]:
    """Every (stem, ending) split of a token with a known ending and a stem of at least 3 letters."""
    return [(token[: len(token) - len(e)], e) for e in ENDINGS if token.endswith(e) and len(token) - len(e) >= 3]


def _match_place(places: dict, window: list[str]) -> Mention | None:
    """The place these tokens name, if exactly one fits. The exact name beats an inflected form of another, then a
    city beats a settlement; places still tied in different raions are skipped."""
    fits = {}
    for combo in _combinations([_splits(w) for w in window]):
        stems, endings = tuple(s for s, _ in combo), tuple(e for _, e in combo)
        for mention, allowed, nominative, priority in places.get(stems, ()):
            if all(e in a for e, a in zip(endings, allowed)):
                rank = (endings != nominative, priority)
                fits[mention] = min(rank, fits.get(mention, rank))
    if not fits:
        return None
    best = min(fits.values())
    top = [m for m, p in fits.items() if p == best]
    return top[0] if len({m.unit_ids for m in top}) == 1 else None


def find_places(text: str | None) -> list[Mention]:
    """Places mentioned in `text`, in order, each once."""
    places, index, phrases = _index()
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
                    match = _match_place(places, window)
                if match:
                    width = n
                    break
        if match and match not in found:
            found.append(match)
        i += width
    return found
