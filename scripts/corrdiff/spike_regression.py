#!/usr/bin/env python3
"""
spike_regression.py -- CorrDiff regression stage vs bilinear ERA5 (tasks/spike_corrdiff.md).

Inputs  (img_lr): 27 ERA5 channels bilinearly regridded onto the 16 x 28 target grid
                  (2.03 km), standardised with training-period statistics.
Target  (img_clean): log1p(hourly mean radar rain), standardised.
Model:  PhysicsNeMo CorrDiffRegressionUNet (SongUNet), trained with CorrDiff's own
        RegressionLoss (MSE of net(zeros, img_lr) against the target).
Baseline: ERA5 total_precipitation (m over the hour -> mm/hr), bilinear to the grid.
Split:  the nowcaster's pinned dates (train < 2 Sep 06:50, val < 14 Sep 03:20,
        test 14 Sep 03:20 .. 25 Sep 23:55 UTC).
Scores on the test period, model vs baseline, with bootstrap 95% CIs over hours:
        RMSE (mm/hr); pooled FSS at 1 and 5 mm/hr in a 4-cell (8 km) window.

Run in the conda env that has PhysicsNeMo:
    <conda sg-weather python> scripts/corrdiff/spike_regression.py [--epochs 60]
Output: results/corrdiff_spike.json, checkpoints/corrdiff_spike/regression.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.radar_dataset import SPLIT_TEST_END, SPLIT_TEST_START, SPLIT_VAL_START  # noqa: E402

TARGETS = ROOT / "data" / "processed" / "corrdiff" / "targets_2km.nc"
ERA5 = ROOT / "data" / "raw" / "era5" / "era5_corrdiff_2026.zarr"
SINGLE = ["u10", "v10", "t2m", "msl", "tcwv", "sp", "tp"]
LEVELS = ["q", "t", "z", "u", "v"]
WIN = 4                                 # FSS window, cells (~8 km)


def build_inputs(t_times, lat, lon):
    """(N, 27, 16, 28) ERA5 on the target grid + the bilinear tp baseline (N, 16, 28) mm/hr."""
    ds = xr.open_zarr(ERA5).sel(time=t_times)
    grid = dict(latitude=xr.DataArray(lat, dims="y"), longitude=xr.DataArray(lon, dims="x"))
    chans, names = [], []
    for v in SINGLE:
        chans.append(ds[v].interp(**grid, method="linear").values)
        names.append(v)
    for v in LEVELS:
        for p in ds.pressure_level.values:
            chans.append(ds[v].sel(pressure_level=p).interp(**grid, method="linear").values)
            names.append(f"{v}{int(p)}")
    x = np.stack(chans, axis=1).astype(np.float32)
    baseline = np.clip(ds["tp"].interp(**grid, method="linear").values * 1000.0, 0, None)   # m/h -> mm/hr
    return x, baseline.astype(np.float32), names


def fss_parts(pred, obs, thr):
    from scipy.ndimage import uniform_filter
    pf = uniform_filter((pred >= thr).astype(np.float64), size=(1, WIN, WIN), mode="constant")
    of = uniform_filter((obs >= thr).astype(np.float64), size=(1, WIN, WIN), mode="constant")
    return ((pf - of) ** 2).sum((1, 2)), (pf ** 2).sum((1, 2)) + (of ** 2).sum((1, 2))


def scores(pred, base, obs, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(obs)
    se_m = ((pred - obs) ** 2).mean((1, 2)); se_b = ((base - obs) ** 2).mean((1, 2))
    out = {"rmse_model": float(np.sqrt(se_m.mean())), "rmse_baseline": float(np.sqrt(se_b.mean()))}
    idx = rng.integers(0, n, (reps, n))
    d = [np.sqrt(se_m[i].mean()) - np.sqrt(se_b[i].mean()) for i in idx]
    out["rmse_diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    for thr in (1.0, 5.0):
        nm, dm = fss_parts(pred, obs, thr); nb, db = fss_parts(base, obs, thr)
        f = lambda num, den, i: 1 - num[i].sum() / max(den[i].sum(), 1e-12)  # noqa: E731
        all_i = np.arange(n)
        diffs = [f(nm, dm, i) - f(nb, db, i) for i in idx]
        out[f"fss{thr:g}"] = {"model": float(f(nm, dm, all_i)), "baseline": float(f(nb, db, all_i)),
                              "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check on whatever ERA5 exists: 70/15/15 split by time, 2 epochs, "
                         "writes nothing to results/")
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tg = xr.open_dataset(TARGETS)
    era_t = xr.open_zarr(ERA5).time.values
    times = np.intersect1d(tg.time.values, era_t)
    rain = tg.rain.sel(time=times).values.astype(np.float32)
    x, base, names = build_inputs(times, tg.lat.values, tg.lon.values)
    ok = np.isfinite(x).all((1, 2, 3)) & np.isfinite(base).all((1, 2))
    times, rain, x, base = times[ok], rain[ok], x[ok], base[ok]

    tm = times.astype("datetime64[m]")
    if args.smoke:
        q = np.quantile(np.arange(len(tm)), [0.7, 0.85]).astype(int)
        tr = np.arange(len(tm)) < q[0]; va = (np.arange(len(tm)) >= q[0]) & (np.arange(len(tm)) < q[1]); te = np.arange(len(tm)) >= q[1]
        args.epochs = 2
    else:
        tr = tm < SPLIT_VAL_START
        va = (tm >= SPLIT_VAL_START) & (tm < SPLIT_TEST_START)
        te = (tm >= SPLIT_TEST_START) & (tm <= SPLIT_TEST_END)
    print(f"hours: {len(times)} matched ({str(tm[0])} .. {str(tm[-1])}); train {tr.sum()}, val {va.sum()}, test {te.sum()}")
    if te.sum() == 0 or tr.sum() == 0:
        sys.exit("ERA5 does not yet cover both the training and the test period; wait for the download")

    mu, sd = x[tr].mean((0, 2, 3), keepdims=True), x[tr].std((0, 2, 3), keepdims=True) + 1e-6
    xs = (x - mu) / sd
    ly = np.log1p(rain)
    ymu, ysd = float(ly[tr].mean()), float(ly[tr].std() + 1e-6)
    ys = ((ly - ymu) / ysd)[:, None]

    from physicsnemo.diffusion.metrics.legacy_losses import RegressionLoss
    from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
    net = CorrDiffRegressionUNet(img_resolution=[16, 28], img_in_channels=x.shape[1], img_out_channels=1,
                                 model_type="SongUNet", model_channels=args.channels, channel_mult=[1, 2, 2],
                                 num_blocks=2).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"CorrDiffRegressionUNet: {n_params / 1e6:.2f} M parameters")
    loss_fn = RegressionLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    X = torch.from_numpy(xs); Y = torch.from_numpy(ys.astype(np.float32))

    def predict(mask):
        net.eval(); out = []
        with torch.no_grad():
            idx = np.flatnonzero(mask)
            for a in range(0, len(idx), 128):
                b = idx[a:a + 128]
                xb = X[b].to(dev)
                out.append(net(torch.zeros(len(b), 1, 16, 28, device=dev), xb).float().cpu().numpy())
        z = np.concatenate(out)[:, 0]
        return np.expm1(z * ysd + ymu).clip(0, None)

    best, best_state, tr_idx = np.inf, None, np.flatnonzero(tr)
    for ep in range(args.epochs):
        net.train(); rng = np.random.default_rng(ep); perm = rng.permutation(tr_idx); tot = 0.0
        for a in range(0, len(perm), args.batch):
            b = perm[a:a + args.batch]
            # fp32: the network is small, and PhysicsNeMo layers refuse autocast
            # unless built with amp_mode=True
            loss = loss_fn(net, Y[b].to(dev), X[b].to(dev)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        pv = predict(va)
        v_rmse = float(np.sqrt(((pv - rain[va]) ** 2).mean()))
        if v_rmse < best:
            best, best_state, best_ep = v_rmse, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} train loss {tot / len(perm):.4f} | val RMSE {v_rmse:.3f} mm/hr (best {best:.3f} @ {best_ep})")
    net.load_state_dict(best_state)
    if args.smoke:
        pt = predict(te)
        print("SMOKE ok:", {k: v for k, v in scores(pt, base[te], rain[te], reps=50).items() if k.startswith("rmse")},
              f"| peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if dev == "cuda" else "",
              f"| {round((time.time() - t0) / 60, 1)} min")
        return
    ckpt = ROOT / "checkpoints" / "corrdiff_spike" / "regression.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, "channels": names,
                "epoch": best_ep}, ckpt)

    pt = predict(te)
    res = scores(pt, base[te], rain[te])
    go = (res["rmse_diff_ci"][1] < 0) and (res["fss1"]["diff_ci"][0] > 0)
    report = {"question": "CorrDiff regression stage vs bilinear ERA5 tp, test period, hourly 2 km rain",
              "hours": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
              "test_period": [str(tm[te][0]), str(tm[te][-1])], "params_M": round(n_params / 1e6, 2),
              "best_epoch": int(best_ep), "val_rmse": best, "test": res,
              "wet_fraction_test_obs": float((rain[te] >= 1).mean()),
              "wet_fraction_test_model": float((pt >= 1).mean()),
              "wet_fraction_test_baseline": float((base[te] >= 1).mean()),
              "verdict": "GO" if go else "NO-GO", "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / "results" / "corrdiff_spike.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
