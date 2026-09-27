#!/usr/bin/env python3
"""spike_diagnostics.py -- why the CorrDiff spike came out NO-GO (tasks/spike_corrdiff.md).

Compares the trained regression (checkpoints/corrdiff_spike/regression.pt) on the test period with
bilinear ERA5 tp, an always-zero forecast and climatology; checks intensity range and the
hour-to-hour correlation of island-mean rain (bootstrap CI). np.corrcoef is avoided: with torch
loaded it crashed natively on this Windows setup, so correlation is computed by hand.
Run in the sg-weather conda env. Output: results/corrdiff_spike_diagnostics.json
"""
import sys, numpy as np, torch, xarray as xr
from pathlib import Path; import os; os.chdir(Path(__file__).resolve().parents[2]); sys.path.insert(0,'scripts/corrdiff'); sys.path.insert(0,'.')
import spike_regression as S
from src.data.radar_dataset import SPLIT_TEST_START, SPLIT_TEST_END
tg=xr.open_dataset(S.TARGETS); t=np.intersect1d(tg.time.values, xr.open_zarr(S.ERA5).time.values)
rain=tg.rain.sel(time=t).values; x,base,_=S.build_inputs(t,tg.lat.values,tg.lon.values)
ok=np.isfinite(x).all((1,2,3))&np.isfinite(base).all((1,2)); t,rain,x,base=t[ok],rain[ok],x[ok],base[ok]
tm=t.astype('datetime64[m]'); te=(tm>=SPLIT_TEST_START)&(tm<=SPLIT_TEST_END)
ck=torch.load('checkpoints/corrdiff_spike/regression.pt',weights_only=False)
from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
net=CorrDiffRegressionUNet(img_resolution=[16,28],img_in_channels=27,img_out_channels=1,model_type='SongUNet',model_channels=64,channel_mult=[1,2,2],num_blocks=2)
net.load_state_dict(ck['model']); net.eval()
xs=torch.from_numpy(((x-ck['mu'])/ck['sd']).astype(np.float32))[te]
with torch.no_grad(): z=net(torch.zeros(len(xs),1,16,28),xs)[:,0].numpy()
pm=np.expm1(z*ck['ysd']+ck['ymu']).clip(0); o=rain[te]; b=base[te]
r=lambda p: float(np.sqrt(((p-o)**2).mean()))
print(f"RMSE mm/hr: model {r(pm):.3f} | ERA5 bilinear {r(b):.3f} | always-zero {r(np.zeros_like(o)):.3f} | climatology {r(np.full_like(o,rain[~te].mean())):.3f}")
print(f"max prediction: model {pm.max():.2f} | ERA5 {b.max():.2f} | observed {o.max():.1f} mm/hr")
dm=lambda a: a.mean((1,2))
def cc(a):
    u, v = dm(a).astype(np.float64), dm(o).astype(np.float64)
    u, v = u - u.mean(), v - v.mean()
    d = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / d) if d > 0 else float('nan')
print(f"hour-to-hour correlation of island-mean rain with observed: model {cc(pm):.2f} | ERA5 {cc(b):.2f}")
wet=dm(o)>0.5
print(f"wet hours (island mean >0.5 mm/hr): {wet.sum()} of {len(o)}; island-mean prediction in wet vs dry hours: model {dm(pm)[wet].mean():.3f} vs {dm(pm)[~wet].mean():.3f}; ERA5 {dm(b)[wet].mean():.3f} vs {dm(b)[~wet].mean():.3f}; observed {dm(o)[wet].mean():.2f}")
rng=np.random.default_rng(0); n=len(o); A,B,O=dm(pm).astype(float),dm(b).astype(float),dm(o).astype(float)
def c(u,v):
    u=u-u.mean(); v=v-v.mean(); d=np.sqrt((u*u).sum()*(v*v).sum()); return (u*v).sum()/d if d>0 else np.nan
ds=[]
for _ in range(2000):
    i=rng.integers(0,n,n); ds.append(c(A[i],O[i])-c(B[i],O[i]))
ds=np.array(ds); ds=ds[np.isfinite(ds)]
print(f"correlation difference model-ERA5: {c(A,O)-c(B,O):+.2f}, 95% CI [{np.percentile(ds,2.5):+.2f}, {np.percentile(ds,97.5):+.2f}]")
import json; json.dump({"rmse": {"model": r(pm), "era5_bilinear": r(b), "always_zero": r(np.zeros_like(o)), "climatology": r(np.full_like(o, rain[~te].mean()))},
  "max_prediction_mm_hr": {"model": float(pm.max()), "era5": float(b.max()), "observed": float(o.max())},
  "island_mean_hourly_correlation": {"model": c(A,O), "era5": c(B,O), "diff_ci": [float(np.percentile(ds,2.5)), float(np.percentile(ds,97.5))]},
  "wet_test_hours": int((O>0.5).sum()), "test_hours": n}, open("results/corrdiff_spike_diagnostics.json","w"), indent=2)
print("-> results/corrdiff_spike_diagnostics.json")
