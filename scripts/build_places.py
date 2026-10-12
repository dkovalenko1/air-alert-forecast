# /// script
# requires-python = ">=3.12"
# dependencies = ["geopandas>=1.0", "pyogrio"]
# ///
"""Build src/airalert/reference/places.csv: cities and settlements with the unit (raion) they lie in (Q12).

Used to match place names in Telegram threat posts to units. Source: OCHA, "Ukraine - Subnational
Administrative Boundaries", admin level 4 (settlements), HDX, CC BY-IGO.

Kept: all cities (463), and settlements (селища) whose name is unique among cities and settlements.
Villages are left out: 52% of village names are shared with another village, so a match would be a guess.
Cities that share a name keep only the entry chosen in CITY_OVERRIDES.

Usage: uv run scripts/build_places.py data/raw/geo/ukr_admin_boundaries.shp.zip
"""

import csv
import pathlib
import sys

import geopandas as gpd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "airalert" / "reference" / "units.csv"
OUT = ROOT / "src" / "airalert" / "reference" / "places.csv"
# Duplicate city names: the one people mean in threat posts.
CITY_OVERRIDES = {"Миколаїв": "Миколаївська", "Городок": None, "Південне": None}  # None: ambiguous, dropped


def key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def main(zip_path: str) -> None:
    with REGISTRY.open(encoding="utf-8") as f:
        registry = list(csv.DictReader(f))
    unit_by_key = {}
    for row in registry:
        for name in [row["name"], *filter(None, row["aliases"].split("|"))]:
            unit_by_key[key(name)] = row["unit_id"]

    g = gpd.read_file(f"zip://{zip_path}!ukr_admin4.shp", ignore_geometry=True)
    g = g[g.adm4_type1.isin(["Місто", "Селище"])].copy()
    g["unit_id"] = [
        "м. Київ" if raion == "Київ" else unit_by_key.get(key(f"{raion} район"))
        for raion in g.adm2_name1
    ]
    g = g[g.unit_id.notna()]
    rows = []
    for name, group in g.groupby("adm4_name1"):
        cities = group[group.adm4_type1 == "Місто"]
        if name in CITY_OVERRIDES:
            oblast = CITY_OVERRIDES[name]
            if oblast is None:
                continue
            group = cities[cities.adm1_name1 == oblast]
        elif len(cities) == 1:
            group = cities
        elif len(group) > 1:
            continue  # settlement name shared with another place: ambiguous
        row = group.iloc[0]
        rows.append({"name": name, "kind": "city" if row.adm4_type1 == "Місто" else "settlement", "unit_id": row.unit_id})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "kind", "unit_id"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["name"]))
    print(f"{len(rows)} places written to {OUT}: {sum(r['kind'] == 'city' for r in rows)} cities, "
          f"{sum(r['kind'] == 'settlement' for r in rows)} settlements")


if __name__ == "__main__":
    main(sys.argv[1])
