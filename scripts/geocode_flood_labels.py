#!/usr/bin/env python3
"""
Geocode PUB flood-label location strings to lat/lon and map them to radar
grid cells, for spatial Stage 5 evaluation.

Only FLOOD_RISK / FLASH_FLOOD labels carry locations; RAIN_WARNINGs are
location-less and skipped. Each raw location string is resolved once and
cached to an editable JSON (data/processed/geocode_cache.json) so the
pipeline is reproducible and works offline after the first run. To fix a
bad match, edit the cache entry by hand and re-run consumers.

Resolution order per string:
  1. cache         (unless --refresh)
  2. curated gazetteer  (offline, hand-verified SG flood locations)
  3. OneMap search API  (online; skipped with --offline)

Usage:
    python scripts/geocode_flood_labels.py            # resolve new strings
    python scripts/geocode_flood_labels.py --offline  # gazetteer + cache only
    python scripts/geocode_flood_labels.py --refresh  # re-resolve everything
    python scripts/geocode_flood_labels.py --status   # report, no network
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "processed" / "flood_labels.parquet"
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
CACHE_PATH = ROOT / "data" / "processed" / "geocode_cache.json"

FLOOD_EVENT_TYPES = {"FLASH_FLOOD", "FLOOD_RISK"}

# Sanity bounds for Singapore (tighter than the radar domain).
SG_BBOX = {"lat_min": 1.15, "lat_max": 1.48, "lon_min": 103.60, "lon_max": 104.05}

ONEMAP_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
ONEMAP_DELAY = 0.4  # be polite to the public endpoint

# Curated gazetteer (lifted from predict_flash_flood/geocode_events.py): known
# SG planning areas + recurring Telegram flood locations. Keys are simplified,
# lowercased road/area names. Offline, hand-verified.
GAZETTEER: dict[str, tuple[float, float]] = {
    "orchard road": (1.3048, 103.8318),
    "bukit timah": (1.3294, 103.8021),
    "bukit timah canal": (1.3400, 103.8050),
    "bukit timah road": (1.3440, 103.7789),
    "maxwell road": (1.2795, 103.8453),
    "ang mo kio": (1.3691, 103.8454),
    "yio chu kang road": (1.3838, 103.8448),
    "toa payoh": (1.3343, 103.8563),
    "hougang": (1.3712, 103.8930),
    "tampines": (1.3496, 103.9568),
    "jurong west": (1.3404, 103.7090),
    "jurong east": (1.3329, 103.7436),
    "woodlands": (1.4367, 103.7864),
    "sengkang": (1.3914, 103.8950),
    "punggol": (1.4043, 103.9021),
    "changi": (1.3644, 103.9915),
    "bedok": (1.3236, 103.9273),
    "yishun": (1.4285, 103.8380),
    "yishun avenue 7": (1.4285, 103.8379),
    "jurong town hall road": (1.3330, 103.7403),
    "pandan road": (1.3183, 103.7456),
    "sims avenue east": (1.3137, 103.9047),
    "upper east coast road": (1.3249, 103.9354),
    "boon lay avenue": (1.3452, 103.7070),
    "enterprise road": (1.3342, 103.7073),
    "upper jurong road": (1.3370, 103.7133),
    "tanjong pagar road": (1.2787, 103.8442),
    "teck whye lane": (1.3607, 103.7597),
    "clementi": (1.3162, 103.7649),
    "pasir ris": (1.3730, 103.9490),
    "choa chu kang": (1.3855, 103.7450),
    "bukit batok": (1.3590, 103.7637),
}

_ABBREV = {
    r"\brd\b": "road", r"\bave\b": "avenue", r"\bav\b": "avenue",
    r"\bst\b": "street", r"\bdr\b": "drive", r"\bjln\b": "jalan",
    r"\bbt\b": "bukit", r"\bupp\b": "upper", r"\blor\b": "lorong",
}


def simplify(loc: str) -> str:
    """Reduce a messy PUB location string to a geocodable road/area name."""
    s = loc
    # 1. Drop time tags "[09:49 hours]" and parentheticals "(near ...)".
    s = re.sub(r"\s*\[[^\]]*\]", "", s)
    s = re.sub(r"\s*\([^)]*\)", "", s)
    # 2. "A off B" -> anchor on B (the named main road, not the slip/service road).
    m = re.search(r"\boff\b\s+(.+)", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    # 3. Strip range / directional qualifiers, keeping the leading road name.
    s = re.sub(r"\s+from\s+.+?\s+to\s+.+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(near|towards|before|between|via)\s+.+", "", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*Singapore$", "", s, flags=re.IGNORECASE)
    # 4. Normalise.
    s = s.strip().lower()
    for pat, full in _ABBREV.items():
        s = re.sub(pat, full, s)
    return re.sub(r"\s+", " ", s).strip()


def in_sg(lat: float, lon: float) -> bool:
    return (SG_BBOX["lat_min"] <= lat <= SG_BBOX["lat_max"] and
            SG_BBOX["lon_min"] <= lon <= SG_BBOX["lon_max"])


def onemap_geocode(query: str) -> dict | None:
    """Query OneMap; return best in-SG result closest to the query name."""
    # OneMap rate-limits bursts with HTTP 429. That used to be treated like "no
    # match", so a busy moment silently left real locations unresolved (found
    # 2026-09-26: 11 of 36 PUB flood-prone areas, all ordinary road names,
    # failed on 429s). Back off and retry instead.
    results = None
    for attempt in range(5):
        try:
            r = requests.get(
                ONEMAP_URL,
                params={"searchVal": query, "returnGeom": "Y",
                        "getAddrDetails": "Y", "pageNum": 1},
                timeout=15,
            )
        except requests.RequestException:
            return None
        if r.status_code == 429:
            time.sleep(2 ** attempt)          # 1, 2, 4, 8, 16 s
            continue
        if r.status_code != 200:
            return None
        try:
            results = r.json().get("results", [])
        except ValueError:
            return None
        break
    if results is None:                       # still rate-limited after retries
        return None

    q = query.lower()
    best, best_score = None, -1.0
    for res in results:
        try:
            lat, lon = float(res["LATITUDE"]), float(res["LONGITUDE"])
        except (KeyError, ValueError, TypeError):
            continue
        if not in_sg(lat, lon):
            continue
        name = (res.get("SEARCHVAL") or "").lower()
        road = (res.get("ROAD_NAME") or "").lower()
        # Prefer results whose name/road contains the query (or vice versa).
        score = 0.0
        if q == name or q == road:
            score = 3.0
        elif q in name or q in road:
            score = 2.0
        elif name in q or road in q:
            score = 1.0
        if score > best_score:
            best, best_score = (lat, lon, res.get("SEARCHVAL")), score
    if best is None:
        return None
    lat, lon, matched = best
    return {"lat": lat, "lon": lon, "matched_name": matched}


def build_grid_mapper():
    """Return fn (lat, lon) -> (lat_idx, lon_idx) for the radar grid."""
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    lats = ds.lat.values.astype(float)
    lons = ds.lon.values.astype(float)
    ds.close()

    def to_idx(lat: float, lon: float) -> tuple[int, int]:
        return int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())

    return to_idx


def collect_location_strings() -> list[str]:
    df = pd.read_parquet(LABELS_PATH)
    flood = df[df["message_type"].isin(FLOOD_EVENT_TYPES)]
    strings: set[str] = set()
    for raw in flood["locations"]:
        for loc in json.loads(raw) if isinstance(raw, str) else (raw or []):
            if loc and loc.strip():
                strings.add(loc.strip())
    return sorted(strings)


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve(loc: str, offline: bool, to_idx) -> dict:
    simp = simplify(loc)
    # 1. gazetteer
    if simp in GAZETTEER:
        lat, lon = GAZETTEER[simp]
        method, matched = "gazetteer", simp
    elif offline:
        return {"lat": None, "lon": None, "method": "unresolved",
                "simplified": simp, "matched_name": None,
                "note": "offline: not in gazetteer; add manually or run online"}
    else:
        # 2. OneMap
        hit = onemap_geocode(simp)
        time.sleep(ONEMAP_DELAY)
        if hit is None:
            return {"lat": None, "lon": None, "method": "unresolved",
                    "simplified": simp, "matched_name": None,
                    "note": "OneMap returned no in-SG match; edit this entry by hand"}
        lat, lon, method, matched = hit["lat"], hit["lon"], "onemap", hit["matched_name"]

    lat_idx, lon_idx = to_idx(lat, lon)
    return {"lat": round(lat, 6), "lon": round(lon, 6),
            "lat_idx": lat_idx, "lon_idx": lon_idx,
            "method": method, "simplified": simp, "matched_name": matched}


def print_summary(cache: dict) -> None:
    total = len(cache)
    resolved = sum(1 for v in cache.values() if v.get("lat") is not None)
    by_method: dict[str, int] = {}
    for v in cache.values():
        by_method[v.get("method", "?")] = by_method.get(v.get("method", "?"), 0) + 1
    print(f"Cache entries : {total}  ({resolved} resolved, {total - resolved} unresolved)")
    for m, n in sorted(by_method.items()):
        print(f"  {m:<11} {n}")
    print(f"Cache file    : {CACHE_PATH.relative_to(ROOT)}")
    for loc, v in cache.items():
        if v.get("lat") is None:
            print(f"  [UNRESOLVED] {loc}  -> simplified '{v.get('simplified')}' "
                  f"({v.get('note','')})")
        else:
            print(f"  [{v['method']:>9}] {loc}  -> ({v['lat']}, {v['lon']}) "
                  f"cell({v['lat_idx']},{v['lon_idx']})  ~ {v.get('matched_name')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Geocode PUB flood-label locations")
    ap.add_argument("--offline", action="store_true", help="Gazetteer + cache only; no OneMap")
    ap.add_argument("--refresh", action="store_true", help="Re-resolve even cached strings")
    ap.add_argument("--status", action="store_true", help="Report cache; no network")
    ap.add_argument("--reindex", action="store_true",
                    help="Recompute lat_idx/lon_idx from cached lat/lon against the current "
                         "radar grid. Use after a grid change; preserves hand-edited "
                         "coordinates (unlike --refresh, which re-resolves them).")
    args = ap.parse_args()

    if args.reindex:
        if not ZARR_PATH.exists():
            print(f"ERROR: {ZARR_PATH.relative_to(ROOT)} not found.")
            sys.exit(1)
        to_idx = build_grid_mapper()
        cache = load_cache()
        moved = 0
        for loc, v in cache.items():
            if v.get("lat") is None:
                continue
            lat_idx, lon_idx = to_idx(v["lat"], v["lon"])
            if (lat_idx, lon_idx) != (v.get("lat_idx"), v.get("lon_idx")):
                v["lat_idx"], v["lon_idx"] = lat_idx, lon_idx
                moved += 1
        save_cache(cache)
        print(f"Re-indexed {moved} entries against the current radar grid "
              f"(coordinates and methods untouched).\n")
        print_summary(cache)
        return

    if args.status:
        cache = load_cache()
        if not cache:
            print("Cache empty. Run without --status to geocode.")
            return
        print_summary(cache)
        return

    for p in (LABELS_PATH, ZARR_PATH):
        if not p.exists():
            print(f"ERROR: {p.relative_to(ROOT)} not found.")
            sys.exit(1)

    to_idx = build_grid_mapper()
    cache = load_cache()
    strings = collect_location_strings()

    new, updated = 0, 0
    for loc in strings:
        if loc in cache and not args.refresh and cache[loc].get("lat") is not None:
            continue
        was_present = loc in cache
        cache[loc] = resolve(loc, args.offline, to_idx)
        updated += was_present
        new += not was_present

    save_cache(cache)
    print(f"Geocoded {len(strings)} flood-label location(s): "
          f"{new} new, {updated} re-resolved.\n")
    print_summary(cache)


if __name__ == "__main__":
    main()
