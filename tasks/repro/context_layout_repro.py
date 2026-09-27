"""
Verification for time-of-day channels + configurable history (2026-09-25).

Why these inputs: the evaluation showed the model's remaining error is where
storms form and die, not how they move (even a shift by the TRUE displacement
lost to persistence). Singapore convection is strongly diurnal and the model had
no clock; 60 min of history lets it see cells building or decaying.

What could go wrong silently, and is checked here:
  1. wrong hour (UTC vs SGT)            -> features match a hand computation
  2. wrong channel order                -> context[-1] is the last radar frame
                                           (residual base + persistence read it)
  3. context_at() drifting from training -> identical to __getitem__
  4. legacy layout changed               -> 6 frames / no time unchanged
  5. old checkpoints unusable            -> 300k model still loads and samples
Expectation: all PASS.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.data.radar_dataset import RadarDataset  # noqa: E402

ok = []
def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<56} {d}"); ok.append(bool(c))

def main():
    ds = RadarDataset("val", context_frames=12, time_channels=True)
    t = ds.indices[len(ds.indices) // 2]
    x = ds[len(ds.indices) // 2]["context"].numpy()

    # 3/2. shape and order
    chk("12 frames + 2 time channels -> 14 channels",
        x.shape[0] == 14 and ds.in_channels == 14, f"shape {x.shape}")
    last = ds._normalise(ds._decode(ds.codes[t - 1]))
    chk("context[-1] is the last radar frame (t-1)",
        np.array_equal(x[-1], last), "residual base / persistence read it")
    first_frame = ds._normalise(ds._decode(ds.codes[t - 12]))
    chk("context[2] is the oldest radar frame (t-12)", np.array_equal(x[2], first_frame))

    # 1. hour, computed independently from the timestamp
    ts = np.datetime64(ds.times[t - 1], "m")
    utc_h = (ts - ts.astype("datetime64[D]")).astype(int) / 60.0
    sgt = (utc_h + 8) % 24
    ang = 2 * np.pi * sgt / 24
    chk("time channels encode SGT hour of last observed frame",
        abs(x[0, 0, 0] - np.sin(ang)) < 1e-6 and abs(x[1, 0, 0] - np.cos(ang)) < 1e-6,
        f"{str(ts)} UTC -> {sgt:05.2f} SGT")
    # max == min, not std() == 0: float32 std of identical values can be a ulp off
    chk("time channels are spatially constant",
        x[0].max() == x[0].min() and x[1].max() == x[1].min(),
        f"std {float(x[0].std()):.1e} (rounding only)")
    chk("sin^2 + cos^2 = 1", abs(x[0, 0, 0]**2 + x[1, 0, 0]**2 - 1) < 1e-6)

    # distinct hours actually differ (guards against a constant feature)
    hours = {round(float(np.arctan2(*ds[i]["context"].numpy()[:2, 0, 0])), 3)
             for i in range(0, len(ds), max(1, len(ds) // 40))}
    chk("feature varies across the day", len(hours) > 10, f"{len(hours)} distinct")

    # 3. context_at == __getitem__
    same = np.array_equal(ds.context_at(t).numpy(), x)
    chk("context_at(t) identical to training input", same)

    # 4. legacy layout unchanged
    leg = RadarDataset("val")                       # defaults: 6 frames, no time
    lt = leg.indices[len(leg.indices) // 2]
    ref = leg._normalise(leg._decode(leg.codes[lt - 6: lt]))
    chk("legacy dataset: 6 frames, no time, unchanged",
        np.array_equal(leg[len(leg.indices) // 2]["context"].numpy(), ref)
        and leg.in_channels == 6)

    # 5. old checkpoint still loads and samples through the eval path
    from evaluate import load_model
    m = load_model(ROOT / "checkpoints/nowcaster/ckpt_step_300000.pt", "cuda")
    # era5_env was added 2026-09-28 (tasks/plan_era5_conditioning.md); a legacy
    # checkpoint must come back with it OFF, like the other legacy options.
    chk("legacy checkpoint -> data_cfg (6, no time, no env)",
        m.data_cfg == {"context_frames": 6, "time_channels": False, "era5_env": False}, str(m.data_cfg))
    with torch.no_grad():
        out = m.ddim_sample(leg.context_at(lt).unsqueeze(0).cuda(), (1, 1, 120, 217))
    chk("legacy checkpoint samples on a legacy context", tuple(out.shape) == (1, 1, 120, 217))

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)

if __name__ == "__main__":
    main()
