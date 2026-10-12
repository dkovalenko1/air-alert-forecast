"""Build src/airalert/reference/units.csv: every raion with its oblast and name variants, plus Kyiv city.

The raion list comes from the official-channel dataset's place list (MIT licence, Vadym Klymenko),
pinned to one commit so the registry is reproducible. Renamed raions keep their old names as aliases,
because older messages and other sources still use them.

Usage: uv run python scripts/build_unit_registry.py
"""

import csv
import json
import pathlib
import urllib.request

STATES_URL = ("https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
              "50ce4820acc9dff56a8aa79ca3cc278aeeeb52a1/processors/states.json")
OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "airalert" / "reference" / "units.csv"

# (oblast, old name) -> current name. Found by matching raion names across our three sources.
RENAMED = {
    ("Дніпропетровська область", "Новомосковський район"): "Самарівський район",
    ("Житомирська область", "Новоград-Волинський район"): "Звягельський район",
    ("Харківська область", "Красноградський район"): "Берестинський район",
    ("Волинська область", "Володимир-Волинський район"): "Володимирський район",
    ("Львівська область", "Червоноградський район"): "Шептицький район",
}
# Permanent alerts: nothing to predict (decision log, DESIGN.md).
EXCLUDED_OBLASTS = {"Луганська область"}


def normalise(name: str) -> str:
    return " ".join(name.replace("’", "'").replace("ʼ", "'").split())


def main() -> None:
    states = json.load(urllib.request.urlopen(STATES_URL))["states"]
    rows = []
    for state in states:
        oblast = normalise(state["stateName"])
        if oblast == "м. Київ":
            rows.append({"unit_id": "м. Київ", "oblast": "м. Київ", "name": "м. Київ", "kind": "city",
                         "aliases": "Київ", "excluded": ""})
            continue
        for district in state["districts"]:
            old = normalise(district["districtName"])
            name = RENAMED.get((oblast, old), old)
            rows.append({
                "unit_id": f"{oblast}/{name}",
                "oblast": oblast,
                "name": name,
                "kind": "raion",
                "aliases": old if old != name else "",
                "excluded": "permanent alert" if oblast in EXCLUDED_OBLASTS else "",
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["unit_id"]))
    print(f"{len(rows)} units written to {OUT}")


if __name__ == "__main__":
    main()
