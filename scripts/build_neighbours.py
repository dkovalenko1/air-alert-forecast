# /// script
# requires-python = ">=3.12"
# dependencies = ["geopandas>=1.0", "pyogrio"]
# ///
"""Build src/airalert/reference/neighbours.csv: pairs of units that share a border (Q11).

Boundaries: OCHA, "Ukraine - Subnational Administrative Boundaries" (admin level 2), HDX, CC BY-IGO,
https://data.humdata.org/dataset/cod-ab-ukr. Only the derived pairs are committed, not the maps.

Usage: uv run scripts/build_neighbours.py data/raw/geo/ukr_admin_boundaries.shp.zip
(geopandas is needed only here, so it is declared inline instead of as a project dependency.)
"""

import csv
import pathlib
import sys

import geopandas as gpd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "airalert" / "reference" / "units.csv"
OUT = ROOT / "src" / "airalert" / "reference" / "neighbours.csv"


def key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def main(zip_path: str) -> None:
    with REGISTRY.open(encoding="utf-8") as f:
        registry = list(csv.DictReader(f))
    by_key = {}
    for row in registry:
        for name in [row["name"], *filter(None, row["aliases"].split("|"))]:
            by_key[key(name)] = row["unit_id"]
    by_key[key("Київ")] = by_key[key("м. Київ")] = "м. Київ"

    raions = gpd.read_file(f"zip://{zip_path}!ukr_admin2.shp")
    name_column = next(c for c in raions.columns if c.lower() in ("adm2_ua", "adm2_name1", "adm2_uk"))
    raions["unit_id"] = raions[name_column].map(lambda n: by_key.get(key(n)) or by_key.get(key(f"{n} район")))
    unmatched = sorted(raions.loc[raions.unit_id.isna(), name_column])
    print(f"{len(raions)} admin-2 areas, {raions.unit_id.notna().sum()} matched to units; unmatched: {unmatched}")

    raions = raions[raions.unit_id.notna()].to_crs(6381)  # Ukraine TM zone, metres
    raions["geometry"] = raions.geometry.buffer(50)  # 50 m tolerance for tiny gaps between neighbouring polygons
    pairs = gpd.sjoin(raions[["unit_id", "geometry"]], raions[["unit_id", "geometry"]], predicate="intersects")
    pairs = sorted({(a, b) for a, b in zip(pairs.unit_id_left, pairs.unit_id_right) if a != b})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "neighbour_id"])
        writer.writerows(pairs)
    counts = {}
    for a, _ in pairs:
        counts[a] = counts.get(a, 0) + 1
    missing = [r["unit_id"] for r in registry if r["unit_id"] not in counts]
    print(f"{len(pairs)} neighbour pairs written to {OUT}; neighbours per unit {min(counts.values())}–{max(counts.values())}; "
          f"units without neighbours: {missing}")


if __name__ == "__main__":
    main(sys.argv[1])
