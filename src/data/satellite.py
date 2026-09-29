"""
satellite.py -- Himawari-9 infrared over Singapore (tasks/plan_satellite.md).

Source: NOAA open data, s3://noaa-himawari9 (public HTTPS, no login). Full-disk AHI Level 1b
every 10 min, 10 horizontal strips per band; Singapore and +-1 deg around it lie inside
strip 5 at 2 km. Each strip is streamed (download -> read -> crop -> delete), because only
a ~50 kB crop per scan is kept and C: has little free space.

Grid: +-1 deg around Singapore on a regular lat/lon grid at 0.018 deg (~2 km), 111 x 111,
row 0 = NORTH (as the radar grid). Values: brightness temperature in K (NaN = missing).

Timing: a scan starting at T is published ~12 min later (08:00 scan -> 08:11:52 on
2026-09-22). `usable_scan(t)` gives the newest scan a forecast issued at t could have used.
Parallax is NOT corrected here: from 140.7 E a 12 km cloud top over Singapore appears
~10 km displaced. The spike measures the offset (scripts/satellite_spike.py).
"""

import bz2
import tempfile
import time as _time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAT_DIR = ROOT / "data" / "processed" / "sat"
BUCKET = "https://noaa-himawari9.s3.amazonaws.com/AHI-L1b-FLDK"
STRIP = 5
BANDS = {"B13": "R20", "B08": "R20"}          # 10.4 um cloud-top temp, 6.2 um water vapour (2 km)
LON0, LON1, LAT0, LAT1, N = 102.85, 104.85, 0.30, 2.30, 111
LAT = np.linspace(LAT1, LAT0, N)              # north first
LON = np.linspace(LON0, LON1, N)
SCAN = np.timedelta64(10, "m")
PUBLISH_DELAY = np.timedelta64(20, "m")       # scan start -> safely available (measured ~12 min)


def _area():
    from pyresample.geometry import AreaDefinition
    d = (LON1 - LON0) / (N - 1) / 2
    return AreaDefinition("sg2km", "Singapore +-1 deg, 0.018 deg", "latlon", "EPSG:4326",
                          N, N, (LON0 - d, LAT0 - d, LON1 + d, LAT1 + d))


def url(t: np.datetime64, band: str) -> str:
    s = str(np.datetime64(t, "m"))                  # 2026-09-22T08:00
    ymd, hm = s[:10].replace("-", ""), s[11:16].replace(":", "")
    return (f"{BUCKET}/{ymd[:4]}/{ymd[4:6]}/{ymd[6:]}/{hm}/"
            f"HS_H09_{ymd}_{hm}_{band}_FLDK_{BANDS[band]}_S{STRIP:02d}10.DAT.bz2")


def fetch(t: np.datetime64, band: str, tries: int = 3) -> bytes | None:
    """The compressed strip, or None if the scan does not exist (404)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(url(t, band), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except OSError as e:
            err = e
        _time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url(t, band)}: {err}")


def crop(raw: bytes, t: np.datetime64, band: str) -> np.ndarray:
    """Compressed HSD strip -> (N, N) float32 brightness temperature on the Singapore grid."""
    from satpy import Scene
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / url(t, band).rsplit("/", 1)[1]            # the reader parses the real file name
        f.with_suffix("").write_bytes(bz2.decompress(raw))
        scn = Scene(reader="ahi_hsd", filenames=[str(f.with_suffix(""))])
        scn.load([band])
        out = scn.resample(_area(), resampler="nearest", radius_of_influence=5000)[band].values
        del scn
    return np.asarray(out, np.float32)


def day_path(band: str, day: str) -> Path:
    return SAT_DIR / band / f"{day}.npz"


def load_days(band: str, days) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate cached days -> (times datetime64[m] of scan START, (T, N, N) float32 K)."""
    ts, xs = [], []
    for d in days:
        p = day_path(band, d)
        if p.exists():
            z = np.load(p)
            ts.append(z["times"].astype("datetime64[m]"))
            xs.append(z["bt"].astype(np.float32))
    if not ts:
        return np.array([], "datetime64[m]"), np.empty((0, N, N), np.float32)
    return np.concatenate(ts), np.concatenate(xs)


def usable_scan(times: np.ndarray, t: np.datetime64) -> int:
    """Index of the newest scan whose start is <= t - PUBLISH_DELAY, or -1."""
    return int(np.searchsorted(times, np.datetime64(t, "m") - PUBLISH_DELAY, side="right")) - 1
