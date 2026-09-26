#!/usr/bin/env python3
"""
build_flood_prone_layer.py -- PUB's official flood-prone areas as a GeoJSON layer.

Source: PUB, "List of Flood Prone Areas in Singapore (as of Nov 2025)", PDF,
https://www.pub.gov.sg/-/media/PUB/PDF/Flood-Management/List-of-Flood-Prone-Areas-as-at-Nov-2025.pdf
(copy kept at data/raw/flood_prone/). data.gov.sg's "Flood Prone Areas" dataset
is only total hectares per year, with no geometry, so the named list is the
closest thing to a map layer PUB publishes.

PUB gives NAMES, not shapes. Each location is simplified to a searchable road or
area and geocoded with OneMap via scripts/geocode_flood_labels.py (same rules and
endpoint the flood-label pipeline uses), producing POINTS:
  - "... at junction of A and B" / "Junction of A and B" -> placed on road A
  - "A / B"                                               -> placed on A
  - "X Area"                                              -> the area name
  - "Building at Road" / "(slip road to X)" on expressways -> the road
Each feature records the query used and how it was placed, so the map does not
claim more precision than it has. Unresolved names are kept with null geometry
and listed, never guessed.

Output: data/processed/flood_prone_areas.geojson
Usage:  python scripts/build_flood_prone_layer.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from geocode_flood_labels import build_grid_mapper, resolve  # noqa: E402

PDF = ROOT / "data" / "raw" / "flood_prone" / "PUB_flood_prone_areas_2025-11.pdf"
OUT = ROOT / "data" / "processed" / "flood_prone_areas.geojson"
SOURCE_URL = ("https://www.pub.gov.sg/-/media/PUB/PDF/Flood-Management/"
              "List-of-Flood-Prone-Areas-as-at-Nov-2025.pdf")


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    """(serial, location) pairs; entries that wrap onto a second line are joined."""
    from pypdf import PdfReader
    lines = []
    for page in PdfReader(str(path)).pages:
        lines += [l.strip() for l in page.extract_text().splitlines()]
    items: list[list] = []
    for line in lines:
        if not line or line.startswith(("List of Flood", "S/N")):
            continue
        m = re.match(r"^(\d{1,3})\s+(.+)$", line)
        if m:
            items.append([int(m.group(1)), m.group(2)])
        elif items:                      # continuation of the previous entry
            items[-1][1] += " " + line
    # PUB uses typographic apostrophes (U+2019/U+2018), which OneMap does not
    # match. (They print as a replacement char on a cp1252 console.)
    return [(n, re.sub(r"\s+", " ", t.replace("’", "'").replace("‘", "'")).strip())
            for n, t in items]


def query_for(loc: str) -> tuple[str, str]:
    """Reduce a PUB location to one geocodable name, and say how."""
    s = loc
    # An expressway is tens of km long; "CTE (slip road to X)" is only useful
    # anchored on X.
    m = re.match(r"^(CTE|PIE|AYE|ECP|KPE|BKE|SLE|TPE|KJE|MCE)\b.*slip road to ([^)]+)", s)
    if m:
        return m.group(2), "expressway slip road: placed on the road it leads to"
    m = re.search(r"junction of (.+?) and (.+)", s, re.IGNORECASE)
    if m:
        return m.group(1), "junction: placed on the first road named"
    if "/" in s:
        return s.split("/")[0], "several roads listed: placed on the first"
    m = re.search(r"\bat\s+(.+)$", s)
    if m:                                    # "People's Association HQ at X Avenue"
        return m.group(1), "building on a road: placed on the road"
    s2 = re.sub(r"\s+Area\b", "", s)
    if s2 != s:
        return s2, "area: placed at the area's geocoded point"
    return s, "direct"


def main() -> None:
    entries = parse_pdf(PDF)
    serials = [n for n, _ in entries]
    if serials != list(range(1, len(serials) + 1)):
        sys.exit(f"PDF parse looks wrong: serials {serials}")
    to_idx = build_grid_mapper()

    import xarray as xr
    z = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)
    lat_rng = (float(z.lat.min()), float(z.lat.max()))
    lon_rng = (float(z.lon.min()), float(z.lon.max()))
    z.close()

    feats, unresolved = [], []
    for sn, loc in entries:
        q, how = query_for(loc)
        r = resolve(q, offline=False, to_idx=to_idx)
        props = {"sn": sn, "location": loc, "query": q, "placement": how,
                 "simplified": r.get("simplified"), "matched_name": r.get("matched_name"),
                 "method": r.get("method")}
        if r.get("lat") is None:
            unresolved.append(f"{sn}: {loc}")
            feats.append({"type": "Feature", "geometry": None, "properties": props})
            continue
        lat, lon = r["lat"], r["lon"]
        props.update({"lat_idx": r["lat_idx"], "lon_idx": r["lon_idx"],
                      "in_radar_domain": lat_rng[0] <= lat <= lat_rng[1]
                      and lon_rng[0] <= lon <= lon_rng[1]})
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": props})
        print(f"  {sn:>2}  {loc[:48]:<48} -> {r['matched_name'] or q}")

    OUT.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "PUB flood-prone areas (Nov 2025), geocoded points",
        "source": SOURCE_URL,
        "note": "Points, not polygons: PUB publishes names only. See 'placement'.",
        "features": feats}, indent=2), encoding="utf-8")
    ok = sum(f["geometry"] is not None for f in feats)
    inside = sum(bool(f["properties"].get("in_radar_domain")) for f in feats)
    print(f"\n{ok}/{len(feats)} located, {inside} inside the radar domain -> {OUT}")
    if unresolved:
        print("UNRESOLVED (null geometry, fix by hand if needed):")
        for u in unresolved:
            print("  " + u)


if __name__ == "__main__":
    main()
