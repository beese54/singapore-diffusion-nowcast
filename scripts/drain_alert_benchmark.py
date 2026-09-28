#!/usr/bin/env python3
"""
drain_alert_benchmark.py -- How many minutes ahead of PUB's drain sensors does the
30-min nowcaster warn? (tasks/plan_phase2.md, Workstream F1)

PUB's "Risk of Flash Floods" alert fires when a drain's water level reaches 90 % of
its depth, so each FLOOD_RISK alert is a drain-sensor measurement with a time and a
place. Only out-of-sample alerts are used (on/after the test-period start; lesson
L031). FLASH_FLOOD reports are scored the same way, separately.

Definitions (fixed before looking at the results):
  site          7 x 7 px box (~2 x 2 km) around the geocoded alert location
  issue time    time of the last radar frame the forecast sees; forecasts issued every
                5 min, valid 35 min after the issue time (the 30-min model)
  model warns   >= 2 of 8 ensemble members have >= 10 mm/hr somewhere in the site box
  radar shows   the OBSERVED frame at that time has >= 10 mm/hr in the site box --
                the "just watch the radar" baseline (persistence)
  lead          alert time - earliest issue time in the 120 min before the alert at
                which the model warned (resp. the radar showed heavy rain)
  held          share of issue times from the first model warning to the alert that
                also warned (1.0 = the warning never dropped)
  head start    model lead - radar lead: the minutes the model adds over the radar
  background    share of the 200 random test forecasts that warned at the same site
                (how often a warning there is "normal"; the chance of a lucky hit)
Only the first alert per site per storm day counts. Storms (days) are the independent
units: summaries are given per storm and over storms, not over alerts.

Usage:  python scripts/drain_alert_benchmark.py [--seed 7000]   (another seed = a seed-sensitivity re-run)
Output: results/drain_alert_benchmark.json
        (forecasts cached in data/processed/eval_cache/drain_bench_<day>.npz)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import load_model  # noqa: E402
from src.data.radar_dataset import (SPLIT_TEST_START, ZARR_PATH,  # noqa: E402
                                    build_context, denormalise_rain, load_stats)

CKPT = ROOT / "checkpoints" / "nowcaster" / "ckpt_step_300000.pt"
TEST_CACHE = ROOT / "data" / "processed" / "eval_cache" / "ens_nowcaster_ckpt_step_300000_s300000_n200_m8.npz"
CACHE = ROOT / "data" / "processed" / "eval_cache"
BOX_R, HEAVY, WARN, WINDOW, MEMBERS = 3, 10.0, 0.25, 120, 8
SGT = pd.Timedelta(hours=8)
STEP = np.timedelta64(5, "m")


def box_max(a, i, j):
    return a[..., max(i - BOX_R, 0):i + BOX_R + 1, max(j - BOX_R, 0):j + BOX_R + 1].max(axis=(-2, -1))


def load_alerts():
    ev = pd.read_parquet(ROOT / "data" / "processed" / "flood_eval_dataset.parquet")
    ev = ev[ev.geocoded & ev.message_type.isin(["FLOOD_RISK", "FLASH_FLOOD"])].copy()
    t = pd.to_datetime(ev.event_datetime.astype(str))
    ev["t"] = t.dt.tz_convert(None) if t.dt.tz is not None else t          # stored as UTC
    ev = ev[ev.t >= pd.Timestamp(SPLIT_TEST_START)].sort_values("t")
    ev["day"] = (ev.t + SGT).dt.strftime("%Y-%m-%d")
    ev["name"] = ev.location_str.astype(str).str.split("[").str[0].str.split().str.join(" ")
    ev["site"] = list(zip(ev.location_lat_idx.astype(int), ev.location_lon_idx.astype(int)))
    return ev.drop_duplicates(["day", "message_type", "site"])             # first alert per site


def forecasts_for_day(model, da, times, day, issue, seed):
    """Ensemble forecasts issued at every time in `issue` (UTC, datetime64[m]); cached."""
    cache = CACHE / f"drain_bench_{day}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["issue"], issue):
            return z["ens"].astype(np.float32)
    off, cf = model.target_offset, model.data_cfg["context_frames"]
    tc = model.data_cfg["time_channels"]
    log_max = max(load_stats()["log_max"], 1e-6)
    # anchor index a = first frame AFTER the context; last seen frame is a - 1
    anchors = np.array([int(np.searchsorted(times, t)) + 1 for t in issue])
    if not all(times[a - 1] == t for a, t in zip(anchors, issue)):
        sys.exit(f"{day}: an issue-time frame is missing from the archive")
    lo = anchors.min() - cf
    if np.any(np.diff(times[lo:anchors.max()]) != STEP):
        sys.exit(f"{day}: archive gap inside the forecast span")
    block = da.isel(time=slice(lo, anchors.max())).values.astype(np.float32)
    if np.isnan(block).any():
        sys.exit(f"{day}: a context frame is blank (failed scrape)")
    dev = next(model.parameters()).device
    ens = np.empty((len(anchors), MEMBERS, *block.shape[1:]), np.float16)
    for k, a in enumerate(anchors):
        ctx = build_context(block[a - cf - lo:a - lo], times[a - 1], log_max, tc,
                            model.data_cfg.get("motion_minutes", 0))
        torch.manual_seed(seed + k)
        with torch.no_grad():
            e = model.ensemble_sample(torch.from_numpy(ctx).unsqueeze(0).to(dev),
                                      n_members=MEMBERS, eta=1.0)
        ens[k] = denormalise_rain(e.cpu().float()[0, :, 0], log_max).numpy()
    np.savez_compressed(cache, ens=ens, issue=issue, lead_steps=off)
    print(f"  {day}: {len(issue)} forecasts cached")
    return ens.astype(np.float32)


def first_lead(flags, issue, t_alert):
    """Minutes from the earliest flagged issue time in the window to the alert."""
    hit = np.flatnonzero(flags)
    if not len(hit):
        return None, None
    k = hit[0]
    held = float(flags[k:].mean())
    return int((t_alert - issue[k]) / np.timedelta64(1, "m")), held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000, help="base random seed of the ensembles")
    seed = ap.parse_args().seed
    alerts = load_alerts()
    da = xr.open_zarr(ZARR_PATH, consolidated=True)["rain_rate"]
    times = da.time.values.astype("datetime64[m]")
    model = load_model(CKPT, "cuda" if torch.cuda.is_available() else "cpu")
    if model.target_offset != 6:
        sys.exit("expected the 30-min model")
    valid_after = np.timedelta64((model.target_offset + 1) * 5, "m")

    test_ens = np.load(TEST_CACHE)["ens"].astype(np.float32)                 # (200, 8, H, W)

    out = {"model": str(CKPT.relative_to(ROOT)), "seed": seed, "definitions": __doc__.split("Definitions")[1].split("Usage")[0].strip(),
           "alerts": [], "storms": {}}
    for day, g in alerts.groupby("day"):
        t0 = np.datetime64(g.t.min(), "m") - np.timedelta64(WINDOW, "m")
        t1 = np.datetime64(g.t.max(), "m")
        t0 -= np.timedelta64(int(t0.astype("int64") % 5), "m")              # onto the 5-min grid
        issue = np.arange(t0, t1 + np.timedelta64(1, "m"), STEP)
        issue = issue[np.isin(issue, times)]                                 # frames that exist
        ens = forecasts_for_day(model, da, times, day, issue, seed)
        obs_idx = np.searchsorted(times, issue)
        obs = da.isel(time=obs_idx).values.astype(np.float32)                # radar at issue time
        valid_idx = np.searchsorted(times, issue + valid_after)
        valid_obs = da.isel(time=valid_idx).values.astype(np.float32)        # what then happened
        for _, r in g.iterrows():
            i, j = r.site
            ta = np.datetime64(r.t, "m")
            win = (issue >= ta - np.timedelta64(WINDOW, "m")) & (issue <= ta)
            p = (box_max(ens[win], i, j) >= HEAVY).mean(1)                    # (n,) share of members
            model_flags = p >= WARN
            radar_flags = box_max(obs[win], i, j) >= HEAVY
            m_lead, held = first_lead(model_flags, issue[win], ta)
            r_lead, _ = first_lead(radar_flags, issue[win], ta)
            bg = float(((box_max(test_ens, i, j) >= HEAVY).mean(1) >= WARN).mean())
            # did heavy rain actually reach the site box within the window (valid times)?
            rained = bool((box_max(valid_obs[win], i, j) >= HEAVY).any())
            out["alerts"].append({
                "day": day, "type": r.message_type, "site": r["name"], "i": i, "j": j,
                "alert_sgt": str(pd.Timestamp(r.t) + SGT)[11:16],
                "model_first_warning_sgt": (str(pd.Timestamp(ta - np.timedelta64(m_lead, "m")) + SGT)[11:16]
                                            if m_lead is not None else None),
                "model_lead_min": m_lead, "model_held": held,
                "radar_lead_min": r_lead,
                "head_start_min": (m_lead - (r_lead or 0)) if m_lead is not None else None,
                "max_member_share": float(p.max()),
                "heavy_rain_observed_at_site": rained,
                "background_warning_rate": bg,
            })
        del ens

    A = pd.DataFrame(out["alerts"])
    for kind in ("FLOOD_RISK", "FLASH_FLOOD"):
        a = A[A.type == kind]
        per = {}
        for day, g in a.groupby("day"):
            w = g[g.model_lead_min.notna()]
            per[day] = {"sites": len(g), "model_warned": len(w),
                        "radar_showed": int(g.radar_lead_min.notna().sum()),
                        "median_model_lead_min": float(w.model_lead_min.median()) if len(w) else None,
                        "median_radar_lead_min": (float(g.radar_lead_min.dropna().median())
                                                  if g.radar_lead_min.notna().any() else None),
                        "median_head_start_min": float(w.head_start_min.median()) if len(w) else None,
                        "median_background_rate": float(g.background_warning_rate.median())}
        out["storms"][kind] = per
        tot = len(a)
        out[f"summary_{kind}"] = {
            "storm_days": len(per), "sites": tot,
            "model_warned_sites": int(a.model_lead_min.notna().sum()),
            "radar_showed_sites": int(a.radar_lead_min.notna().sum()),
            "median_model_lead_min": float(a.model_lead_min.median()) if a.model_lead_min.notna().any() else None,
            "median_radar_lead_min": float(a.radar_lead_min.median()) if a.radar_lead_min.notna().any() else None,
            "storms_with_positive_median_head_start": sum(
                1 for s in per.values() if (s["median_head_start_min"] or 0) > 0),
        }

    out["caveats"] = [
        "Three storm days (18, 22, 27 Sep 2026): a small sample; per-alert numbers are not independent.",
        "Warnings are counted in a 2 x 2 km box around the geocoded road name, not the drain sensor itself.",
        "Earliest warning in a 120-min window can come from an earlier, unrelated shower; 'held' shows "
        "whether the warning persisted to the alert.",
        "Forecast issue assumes the radar frame is available immediately; NEA publishes frames a few "
        "minutes late, so real leads are a few minutes shorter.",
    ]
    path = ROOT / "results" / ("drain_alert_benchmark.json" if seed == 7000
                               else f"drain_alert_benchmark_seed{seed}.json")
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(A[["day", "type", "site", "alert_sgt", "model_lead_min", "model_held", "radar_lead_min",
             "head_start_min", "background_warning_rate"]].to_string(index=False))
    for kind in ("FLOOD_RISK", "FLASH_FLOOD"):
        print(kind, json.dumps(out[f"summary_{kind}"]))
        for d, s in out["storms"][kind].items():
            print("  ", d, json.dumps(s))
    print("->", path)


if __name__ == "__main__":
    main()
