#!/usr/bin/env python3
"""
build_dashboard.py -- Export the static dashboard's data (dashboard/data/).

The dashboard (dashboard/index.html) is a static site: no server, no GPU.
Everything it shows is exported here from the caches and results this repo
already has, so every number and image on it traces back to a file.

Rain fields are packed as GREYSCALE PNG sprite sheets (one grid cell per frame,
native 217 x 120 px) and coloured in the browser with a lookup table. One byte
per pixel:
    rain       byte = round(255 * log1p(mm/hr) / log1p(100))   (0 = dry)
    probability byte = round(255 * p)
The radar is ~97% dry, so these compress to a few KB per frame.

Inputs:  data/processed/radar.zarr, data/processed/eval_cache/case_<name>_lead*.npz, results/warning_skill.json,
         results/warning_calibration.json, results/hybrid_warning.json,
         data/processed/flood_eval_dataset.parquet, data/processed/flood_prone_areas.geojson,
         results/*.json, checkpoints/nowcaster/ckpt_step_300000.pt (denoising demo, GPU optional)
Output:  dashboard/data/*.png, dashboard/data/data.json, dashboard/data/radar240/*.png

Usage:   python scripts/build_dashboard.py
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_model  # noqa: E402
from src.data.radar_dataset import build_context, denormalise_rain, load_stats  # noqa: E402

OUT = ROOT / "dashboard" / "data"
SGT = np.timedelta64(8, "h")
LOGMAX = math.log1p(100.0)
BOX_R = 3                       # 7 x 7 px (~2 x 2 km) box around a flood site

# Cases: observed window (UTC) wide enough to show persistence for the 90-min
# lead, forecast targets from case_study_forecasts.py, and the flood site.
CASES = {
    "22sep": dict(title="22 Sep 2026 — King's Road / Coronation Road",
                  obs=("2026-09-22T05:00", "2026-09-22T11:00"),
                  site=(52, 93), site_name="King's Road",
                  denoise_target="2026-09-22T08:50"),
    "27sep": dict(title="27 Sep 2026 — Neo Pee Teck Lane / Pasir Panjang Road",
                  obs=("2026-09-27T00:30", "2026-09-27T05:30"),
                  site=(63, 78), site_name="Neo Pee Teck Lane / Pasir Panjang Rd"),
}


def rain_bytes(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, np.float32), nan=0.0).clip(0, 100)
    return np.round(255 * np.log1p(a) / LOGMAX).astype(np.uint8)


def prob_bytes(p: np.ndarray) -> np.ndarray:
    return np.round(255 * np.asarray(p, np.float32).clip(0, 1)).astype(np.uint8)


def save_sprite(frames: np.ndarray, name: str, cols: int = 10) -> dict:
    """(N, H, W) uint8 -> greyscale PNG grid; returns its layout for the page."""
    n, h, w = frames.shape
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * h, cols * w), np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[k]
    Image.fromarray(sheet, mode="L").save(OUT / name, optimize=True)
    return {"file": f"data/{name}", "n": n, "cols": cols, "w": w, "h": h}


def box_max(a: np.ndarray, site) -> np.ndarray:
    i, j = site
    return a[..., i - BOX_R:i + BOX_R + 1, j - BOX_R:j + BOX_R + 1].max(axis=(-2, -1))


def hhmm(t) -> str:
    return str((np.datetime64(t, "m") + SGT))[11:16]


def coastline(lat, lon) -> list[str]:
    """Singapore land outline as SVG path strings in pixel coordinates."""
    import cartopy.feature as cfeature
    from shapely.geometry import box
    x0, x1, y0, y1 = lon.min(), lon.max(), lat.min(), lat.max()
    W, H = len(lon), len(lat)
    clip = box(x0 - 0.05, y0 - 0.05, x1 + 0.05, y1 + 0.05)
    paths = []
    for geom in cfeature.NaturalEarthFeature("physical", "land", "10m").intersecting_geometries(
            [x0 - 0.05, x1 + 0.05, y0 - 0.05, y1 + 0.05]):
        g = geom.intersection(clip)
        polys = getattr(g, "geoms", [g])
        for poly in polys:
            if poly.is_empty or not hasattr(poly, "exterior"):
                continue
            pts = [((x - x0) / (x1 - x0) * (W - 1), (y1 - y) / (y1 - y0) * (H - 1))
                   for x, y in poly.exterior.coords]
            paths.append("M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts) + " Z")
    return paths


def to_px(lat_v, lon_v, lat, lon):
    x0, x1, y0, y1 = lon.min(), lon.max(), lat.min(), lat.max()
    return (round((lon_v - x0) / (x1 - x0) * (len(lon) - 1), 1),
            round((y1 - lat_v) / (y1 - y0) * (len(lat) - 1), 1))


def export_case(name: str, cfg: dict, da: xr.DataArray, ev: pd.DataFrame, lat, lon) -> dict:
    obs = da.sel(time=slice(*cfg["obs"])).load()
    otimes = obs.time.values
    out = {"title": cfg["title"], "site": {"name": cfg["site_name"],
                                           "i": cfg["site"][0], "j": cfg["site"][1], "r": BOX_R},
           "obs_times": [hhmm(t) for t in otimes],
           "obs": save_sprite(rain_bytes(obs.values), f"{name}_obs.png"),
           "obs_site": [round(float(v), 1) for v in box_max(obs.values, cfg["site"])],
           "leads": {}}
    for lead in (30, 60, 90):
        p = ROOT / "data" / "processed" / "eval_cache" / f"case_{name}_lead{lead}.npz"
        if not p.exists():
            print(f"  {name}: no lead-{lead} cache, skipped")
            continue
        z = np.load(p)
        ens = z["ens"].astype(np.float32)                      # (N, M, H, W)
        off = int(z["lead_steps"])
        tt = z["target_times"].astype("datetime64[ns]")
        m = box_max(ens, cfg["site"])                          # (N, M)
        out["leads"][str(lead)] = {
            "offset_steps": off,
            "minutes_after_last_frame": (off + 1) * 5,
            "targets": [hhmm(t) for t in tt],
            "issued": [hhmm(t - np.timedelta64((off + 1) * 5, "m")) for t in tt],
            # index into obs_times of each target and of its persistence frame
            "target_obs_index": [int(np.searchsorted(otimes, t)) for t in tt],
            "persistence_obs_index": [int(np.searchsorted(otimes, t - np.timedelta64((off + 1) * 5, "m")))
                                      for t in tt],
            "mean": save_sprite(rain_bytes(ens.mean(1)), f"{name}_lead{lead}_mean.png"),
            "p10": save_sprite(prob_bytes((ens >= 10).mean(1)), f"{name}_lead{lead}_p10.png"),
            "member": save_sprite(rain_bytes(ens[:, 0]), f"{name}_lead{lead}_member.png"),
            "site_median": [round(float(v), 1) for v in np.median(m, 1)],
            "site_min": [round(float(v), 1) for v in m.min(1)],
            "site_max": [round(float(v), 1) for v in m.max(1)],
            "site_p10": [round(float(v), 3) for v in (m >= 10).mean(1)],
            "site_p30": [round(float(v), 3) for v in (m >= 30).mean(1)],
        }
    day = cfg["obs"][0][:10]
    e = ev[ev.event_datetime.astype(str).str.startswith(day)
           & ev.message_type.isin(["FLASH_FLOOD", "FLOOD_RISK"]) & ev.geocoded]
    out["reports"] = []
    for _, r in e.sort_values("event_datetime").iterrows():
        x, y = to_px(r.location_lat, r.location_lon, lat, lon)
        ts = pd.Timestamp(r.event_datetime)
        ts = ts.tz_convert(None) if ts.tzinfo else ts          # stored as UTC
        out["reports"].append({"time": hhmm(np.datetime64(ts)),
                               "type": r.message_type,
                               "name": " ".join(str(r.location_str).split("[")[0].split()),
                               "x": x, "y": y})
    return out


def export_denoise(da: xr.DataArray, target: str) -> dict:
    """One member's 50 DDIM steps for a 22 Sep forecast: x_t and the running
    clean-sample estimate, the heart of 'noise -> rain map'."""
    ckpt = ROOT / "checkpoints" / "nowcaster" / "ckpt_step_300000.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, dev)
    off, cf = model.target_offset, model.data_cfg["context_frames"]
    t_target = np.datetime64(target, "ns")
    issue = t_target - np.timedelta64((off + 1) * 5, "m")
    frames = da.sel(time=slice(issue - np.timedelta64((cf - 1) * 5, "m"), issue)).values.astype(np.float32)
    assert frames.shape[0] == cf, frames.shape
    log_max = max(load_stats()["log_max"], 1e-6)
    ctx = torch.from_numpy(build_context(frames, issue, log_max, model.data_cfg["time_channels"],
                                         model.data_cfg.get("motion_minutes", 0)))
    xs, x0s = [], []

    def cb(i, x, x0):
        xs.append(denormalise_rain(x.clamp(-1, 1).cpu().float()[0, 0], log_max).numpy())
        x0s.append(denormalise_rain(x0.cpu().float()[0, 0], log_max).numpy())

    torch.manual_seed(2026)
    with torch.no_grad():
        model.ddim_sample(ctx.unsqueeze(0).to(dev), (1, 1, *frames.shape[1:]), eta=1.0, callback=cb)
    return {"target": hhmm(t_target), "issued": hhmm(issue),
            "steps": len(xs),
            "context": save_sprite(rain_bytes(frames), "denoise_context.png", cols=6),
            "x_t": save_sprite(rain_bytes(np.stack(xs)), "denoise_xt.png"),
            "x0": save_sprite(rain_bytes(np.stack(x0s)), "denoise_x0.png"),
            "observed": save_sprite(rain_bytes(da.sel(time=t_target).values[None]), "denoise_obs.png", cols=1)}


def export_results() -> dict:
    E = json.loads((ROOT / "results" / "evaluation_report.json").read_text())["by_lead"]
    res = {"headline": {}, "fss_by_scale": {}, "flood_events": {}, "period": None}
    for L, r in E.items():
        f = r["fss_pooled_ensemble_probability"]
        res["headline"][L] = {
            "crps_skill": r["crps_skill"], "crps_ci": r["crps_skill_95ci"],
            "fss2": [f["2.0 mm/hr"]["11.9 km"]["model"], f["2.0 mm/hr"]["11.9 km"]["persistence"]],
            "fss2_ci": f["2.0 mm/hr"]["11.9 km"]["diff_95ci"],
            "fss10": [f["10.0 mm/hr"]["11.9 km"]["model"], f["10.0 mm/hr"]["11.9 km"]["persistence"]],
            "fss10_ci": f["10.0 mm/hr"]["11.9 km"]["diff_95ci"]}
    for L in (30, 60, 90):
        d = json.loads((ROOT / "results" / f"probabilistic_eval_lead{L}.json").read_text())
        res["period"] = d["test_period"]
        res["fss_by_scale"][str(L)] = {thr: {"model": d["A_fss"][thr]["model: ensemble probability"],
                                             "persistence": d["A_fss"][thr]["persistence"]}
                                       for thr in ("0.5", "2.0", "10.0")}
        D = d["D_flood_events"]
        res["flood_events"][str(L)] = {g: {k: v for k, v in D[g][">=2.0 mm/hr within 1.4 km"].items()}
                                       | {"n": D[g]["n_events"]} for g in ("test", "after_test")}
        res["flood_events"][str(L)]["excluded_in_sample"] = D["excluded_in_sample"]
    # plain-language warning skill (scripts/warning_skill.py)
    res["warning"] = json.loads((ROOT / "results" / "warning_skill.json").read_text())["by_lead"]
    # recalibrated 60-min rule and the hybrid comparison (scripts/warning_calibration.py, hybrid_warning.py)
    cal = json.loads((ROOT / "results" / "warning_calibration.json").read_text())["by_lead"]["60"]
    hyb = json.loads((ROOT / "results" / "hybrid_warning.json").read_text())
    res["warning60"] = {"rule": cal["chosen"], "verdict": cal["verdict"],
                        "cal": cal["test"]["calibrated"], "uncal": cal["test"]["uncalibrated"],
                        "extrap": hyb["scores"]["extrap"], "hybrid": hyb["scores"]["H"],
                        "hybrid_verdict": hyb["verdict"]}
    return res


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    da = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)["rain_rate"]
    lat, lon = da.lat.values, da.lon.values
    ev = pd.read_parquet(ROOT / "data" / "processed" / "flood_eval_dataset.parquet")

    data = {"grid": {"w": len(lon), "h": len(lat), "km_per_px": 0.290,
                     "lat": [float(lat.min()), float(lat.max())],
                     "lon": [float(lon.min()), float(lon.max())]},
            "coast": coastline(lat, lon),
            "cases": {}, "results": export_results()}
    for name, cfg in CASES.items():
        print(f"case {name}")
        data["cases"][name] = export_case(name, cfg, da, ev, lat, lon)
    print("denoising demo")
    data["denoise"] = export_denoise(da, CASES["22sep"]["denoise_target"])

    gj = json.loads((ROOT / "data" / "processed" / "flood_prone_areas.geojson").read_text(encoding="utf-8"))
    data["flood_prone"] = []
    for f in gj["features"]:
        if f["geometry"] and f["properties"].get("in_radar_domain"):
            x, y = to_px(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0], lat, lon)
            data["flood_prone"].append({"sn": f["properties"]["sn"], "name": f["properties"]["location"],
                                        "x": x, "y": y})

    # 240 km context loop for 22 Sep (raw NEA images, shown with attribution)
    src = ROOT / "data" / "raw" / "radar_240km"
    (OUT / "radar240").mkdir()
    r240 = []
    for t in np.arange(np.datetime64("2026-09-22T06:30"), np.datetime64("2026-09-22T09:31"),
                       np.timedelta64(10, "m")):
        f = src / (str(t).replace("-", "").replace("T", "_").replace(":", "")[:13] + ".png")
        if f.exists():
            shutil.copy(f, OUT / "radar240" / f.name)
            r240.append({"file": f"data/radar240/{f.name}", "time": hhmm(t)})
    data["radar240"] = r240
    print(f"240 km frames: {len(r240)}")

    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"dashboard/data: {sum(1 for _ in OUT.rglob('*.png'))} PNGs, {total / 1e6:.2f} MB total")


if __name__ == "__main__":
    main()
