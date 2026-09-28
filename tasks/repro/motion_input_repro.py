"""
Verification for the optional motion input channel (2026-09-28, tasks/plan_motion_input.md).

What must hold:
- OFF by default: the context tensor, channel count and legacy checkpoints are unchanged.
- ON, warm-started from a model without it: the new stem slice is zero, so the output
  equals the source model's for ANY motion channel -- training starts from the known-good
  60-min model and any change is learned; and that slice receives a gradient.
- No leakage: the channel for anchor t is unchanged when frames t onward are replaced.
- The channel is the motion forecast of the context frames; the motion code moves a
  known synthetic shift to the right place.
- Guards: resume refuses a motion mismatch; warm start refuses N -> 0; the eval loader
  rebuilds a motion model from its stamp.

Expectation: all PASS. Needs checkpoints/ and data/processed/; CPU is fine.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.motion import motion_forecast  # noqa: E402
from src.data.radar_dataset import RadarDataset, normalise_rain  # noqa: E402
from src.model.diffusion import GaussianDiffusion  # noqa: E402
from src.model.unet import ConditionedUNet  # noqa: E402

BASE = ROOT / "checkpoints" / "nowcaster" / "lead60_warm" / "ckpt_step_100000.pt"
MM = 65
ok = []


def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<64} {d}")
    ok.append(bool(c))


def build(in_ch):
    return GaussianDiffusion(ConditionedUNet(context_frames=in_ch, use_checkpoint=False), parameterization="v")


def main():
    import train as tr
    state = torch.load(BASE, map_location="cpu", weights_only=True)

    # 1. off by default
    off = RadarDataset("test", target_offset=12)
    on = RadarDataset("test", target_offset=12, motion_minutes=MM)
    t = off.indices[len(off.indices) // 2]
    frames = off._decode(off.codes[t - 6:t])
    chk("off: 6 channels, context = normalised frames",
        off.in_channels == 6 and np.array_equal(off.context_at(t).numpy(), normalise_rain(frames, off.log_max)))
    base = build(6)
    base.load_state_dict(state["model"])                       # strict
    chk("off: 60-min checkpoint loads strictly", True)

    # 5. the channel is the motion forecast of the context frames, prepended
    c_on = on.context_at(t).numpy()
    chk("on: 7 channels, [motion, frames]; frames unchanged",
        on.in_channels == 7 and np.array_equal(c_on[1:], off.context_at(t).numpy()))
    chk("on: channel 0 = normalised motion forecast (65 min)",
        np.array_equal(c_on[0], normalise_rain(motion_forecast(frames, MM), on.log_max)))

    # 4. no leakage: replace frames t .. t+30 with random codes, context unchanged
    codes = on.codes
    was_writeable = codes.flags.writeable
    codes.flags.writeable = True
    saved = codes[t:t + 30].copy()
    try:
        codes[t:t + 30] = np.random.default_rng(0).integers(0, 30, saved.shape, dtype=codes.dtype)
        chk("no leakage: context unchanged when frames >= t change", np.array_equal(on.context_at(t).numpy(), c_on))
    finally:
        codes[t:t + 30] = saved
        codes.flags.writeable = was_writeable

    # synthetic motion check on the 70 km grid
    yy, xx = np.mgrid[0:120, 0:217]
    g = lambda cx, cy: np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 72.0)  # noqa: E731
    blob = lambda cx, cy: (30 * g(cx, cy) * (g(cx, cy) > 0.05)).astype(np.float32)  # noqa: E731
    stack = np.stack([blob(60 + 4 * k, 70 - k) for k in range(6)])  # (+4, -1) px per 5 min
    f = motion_forecast(stack, MM)
    m = f > 1
    cx, cy = xx[m].mean(), yy[m].mean()
    chk("motion: synthetic shift lands where expected (132, 52)", abs(cx - 132) < 3 and abs(cy - 52) < 3,
        f"got ({cx:.1f}, {cy:.1f})")

    # 2. warm start: zero stem slice -> identical output for any motion channel
    cond = build(7)
    tr.init_weights_from(BASE, cond, "v", False, (6, False), MM)
    w = cond.model.stem.weight
    chk("warm start: stem slice for the motion channel is exactly 0", float(w[:, 1].abs().max()) == 0.0)
    torch.manual_seed(0)
    x = torch.randn(2, 1, 120, 217)
    ctx = torch.rand(2, 6, 120, 217) * 2 - 1
    tt = torch.tensor([10, 900])
    base.eval(); cond.eval()
    with torch.no_grad():
        yb = base.model(x, ctx, tt)
        ya = cond.model(x, torch.cat([torch.rand(2, 1, 120, 217) * 2 - 1, ctx], 1), tt)
    chk("warm start at step 0: output == 60-min model (random motion)", torch.equal(ya, yb),
        f"max diff {float((ya - yb).abs().max()):.2e}")

    # 3. it can learn
    cond.train()
    loss = cond.p_losses(torch.rand(2, 1, 120, 217) * 2 - 1, torch.rand(2, 7, 120, 217) * 2 - 1)
    loss.backward()
    gs = cond.model.stem.weight.grad[:, 1]
    chk("motion stem slice receives a non-zero gradient", float(gs.abs().sum()) > 0)

    # 6. guards and the eval loader
    with tempfile.TemporaryDirectory() as d:
        pm = Path(d) / "motion.pt"
        torch.save({**state, "model": cond.state_dict(), "motion_minutes": MM}, pm)
        m7 = build(7); opt = torch.optim.AdamW(m7.parameters())
        try:
            tr.load_checkpoint(pm, m7, opt, torch.amp.GradScaler("cuda", enabled=False), "v", False, (6, False), 0)
            chk("resume refuses motion_minutes mismatch", False, "no error")
        except SystemExit:
            chk("resume refuses motion_minutes mismatch", True)
        try:
            tr.init_weights_from(pm, build(6), "v", False, (6, False), 0)
            chk("warm start refuses motion N -> 0", False, "no error")
        except SystemExit:
            chk("warm start refuses motion N -> 0", True)
        from evaluate import load_model
        lm = load_model(pm, "cpu")
        chk("eval loader rebuilds the motion model from its stamp",
            lm.data_cfg.get("motion_minutes") == MM and lm.model.stem.weight.shape[1] == 8)

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
