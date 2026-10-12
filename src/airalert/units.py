"""The units we forecast (raions and Kyiv city) and how places named in alert messages map onto them.

The registry lives in reference/units.csv (built by scripts/build_unit_registry.py). Each alert place maps
onto units like this (DESIGN.md, section 2):

- a raion → that raion;
- an oblast → every raion of that oblast (oblast-wide alerts cover all its raions);
- Kyiv city → its own unit;
- any other city zone → no unit: city zones are ignored unless Q16 promotes them to units, and the
  raion's own alerts stand for them.
"""

import csv
import functools
import pathlib
from dataclasses import dataclass

from airalert.alarm_signal import Location

REGISTRY = pathlib.Path(__file__).resolve().parent / "reference" / "units.csv"


@dataclass(frozen=True)
class Unit:
    unit_id: str
    oblast: str
    name: str
    kind: str
    excluded: str  # reason, or "" if the unit is forecast
    aliases: tuple[str, ...] = ()  # former names, e.g. Новомосковський район for Самарівський район


def _normalise(name: str) -> str:
    return " ".join(name.replace("’", "'").replace("ʼ", "'").split())


def _key(name: str) -> str:
    """Loose match key: '#Кам'янський_район', 'Кам’янський район' and 'камянський район' all match."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _oblast_name(name: str) -> str:
    """'Харківська обл.' and 'Харківська область' both become 'Харківська область'."""
    name = _normalise(name)
    return name[: -len(" обл.")] + " область" if name.endswith(" обл.") else name


@functools.cache
def units() -> tuple[Unit, ...]:
    with REGISTRY.open(encoding="utf-8") as f:
        return tuple(Unit(r["unit_id"], r["oblast"], r["name"], r["kind"], r["excluded"], tuple(filter(None, r["aliases"].split("|"))))
                     for r in csv.DictReader(f))


def forecast_unit_ids() -> list[str]:
    """The fixed set of units every grid covers, whether or not they ever had an alert (excluded ones left out)."""
    return [u.unit_id for u in units() if not u.excluded]


@functools.cache
def _index() -> tuple[dict, dict]:
    by_name, by_oblast = {}, {}
    with REGISTRY.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for name in [r["name"], *filter(None, r["aliases"].split("|"))]:
                by_name[(r["oblast"], _key(name))] = r["unit_id"]
                by_name[(None, _key(name))] = r["unit_id"]  # raion names are unique nationwide
            if r["kind"] == "raion":
                by_oblast.setdefault(r["oblast"], []).append(r["unit_id"])
    return by_name, by_oblast


class UnknownPlace(ValueError):
    pass


def canonical(location: Location) -> Location:
    """The same place under one fixed name, so that alerts and all-clears pair up even when the channel switches
    between a raion's old and new name or spells it differently. Levels are kept: an oblast-wide alert stays
    oblast-wide. City zones other than Kyiv are returned unchanged (they map to no unit). Raises UnknownPlace."""
    if location.level == "city":
        return Location("city", "Київ", None) if resolve(location) else location
    unit_ids = resolve(location)
    if location.level == "oblast":
        oblast = next(u.oblast for u in units() if u.unit_id == unit_ids[0])
        return Location("oblast", oblast, oblast)
    unit = next(u for u in units() if u.unit_id == unit_ids[0])
    return Location("raion", unit.name, unit.oblast)


def resolve(location: Location) -> list[str]:
    """Unit ids an alert at `location` covers. Raises UnknownPlace for names not in the registry."""
    by_name, by_oblast = _index()
    if location.level == "city":
        return ["м. Київ"] if _key(location.name) in ("київ", "мкиїв") else []
    oblast = _oblast_name(location.oblast) if location.oblast else None
    if location.level == "oblast":
        oblast = next((o for o in by_oblast if _key(o) == _key(oblast or "")), oblast)
        if oblast not in by_oblast:
            raise UnknownPlace(oblast)
        return list(by_oblast[oblast])
    unit_id = by_name.get((oblast, _key(location.name)))
    if unit_id is None:
        raise UnknownPlace(f"{location.name} ({oblast})")
    return [unit_id]
