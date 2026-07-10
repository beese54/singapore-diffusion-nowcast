#!/usr/bin/env python3
"""
status.py — Single-command progress report for the SG weather diffusion project.

Usage:
    python scripts/status.py
"""

import json
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
DATA = ROOT / "data"

RADAR_START = date(2026, 5, 22)
RADAR_GOAL_DAYS = 90
RADAR_INTERVAL_MIN = 5
EXPECTED_PER_DAY = 24 * 60 // RADAR_INTERVAL_MIN  # 288


def flag(name: str) -> bool:
    return (CHECKPOINTS / f"{name}_complete.flag").exists()


def check_radar() -> dict:
    png_dir = DATA / "raw" / "radar"
    zarr_path = DATA / "processed" / "radar.zarr"

    png_count = len(list(png_dir.glob("*.png"))) if png_dir.exists() else 0

    zarr_count = 0
    zarr_first = zarr_last = None
    if zarr_path.exists():
        try:
            import xarray as xr
            ds = xr.open_zarr(zarr_path, consolidated=True)
            times = ds.time.values
            zarr_count = len(times)
            if zarr_count:
                import pandas as pd
                zarr_first = pd.Timestamp(times[0]).to_pydatetime()
                zarr_last = pd.Timestamp(times[-1]).to_pydatetime()
            ds.close()
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()
    days_collected = (today - RADAR_START).days
    pct = min(100, round(days_collected / RADAR_GOAL_DAYS * 100, 1))
    days_remaining = max(0, RADAR_GOAL_DAYS - days_collected)
    from datetime import timedelta
    eta = RADAR_START + timedelta(days=RADAR_GOAL_DAYS)

    return {
        "png_count": png_count,
        "zarr_count": zarr_count,
        "zarr_first": zarr_first,
        "zarr_last": zarr_last,
        "days_collected": days_collected,
        "days_remaining": days_remaining,
        "pct": pct,
        "eta": eta,
        "complete": flag("stage3"),
    }


def check_flood_labels() -> dict:
    messages_path = DATA / "raw" / "flood_labels" / "messages.json"
    state_path = DATA / "raw" / "flood_labels" / "state.json"
    parquet_path = DATA / "processed" / "flood_labels.parquet"
    session_exists = (CHECKPOINTS / "telegram.session").exists()

    msg_count = 0
    msg_first = msg_last = None
    if messages_path.exists():
        msgs = json.loads(messages_path.read_text(encoding="utf-8"))
        msg_count = len(msgs)
        if msgs:
            dates = sorted(m["date"] for m in msgs)
            msg_first = dates[0][:10]
            msg_last = dates[-1][:10]

    last_sync = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        last_sync = state.get("last_run", "")[:16]

    type_counts: dict[str, int] = {}
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            type_counts = df["message_type"].value_counts().to_dict()
        except Exception:
            pass

    return {
        "session": session_exists,
        "msg_count": msg_count,
        "msg_first": msg_first,
        "msg_last": msg_last,
        "last_sync": last_sync,
        "type_counts": type_counts,
    }


def bar(pct: float, width: int = 30) -> str:
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct}%"


def main() -> None:
    now = datetime.now(timezone.utc)
    print(f"\nSG Weather Diffusion - Status Report  ({now.strftime('%Y-%m-%d %H:%M')} UTC)")
    print("=" * 62)

    # Stages 0-2
    stages = [
        ("Stage 0", "Environment setup",          flag("stage0")),
        ("Stage 1", "ERA5 download",               flag("stage1")),
        ("Stage 2", "PrecipitationAFNO baseline",  flag("stage2")),
    ]
    for name, desc, done in stages:
        status = "COMPLETE" if done else "PENDING"
        mark = "+" if done else " "
        print(f"  [{mark}] {name}  {desc:<32}  {status}")

    print()

    # Stage 3 — radar archive
    r = check_radar()
    mark = "+" if r["complete"] else " "
    print(f"  [{mark}] Stage 3  Radar archive")
    print(f"         {bar(r['pct'])}")
    print(f"         {r['days_collected']}/{RADAR_GOAL_DAYS} days  |  {r['zarr_count']:,} zarr frames  |  {r['png_count']:,} PNGs")
    if r["zarr_last"]:
        print(f"         Latest frame : {r['zarr_last'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"         ETA 90 days  : {r['eta'].strftime('%Y-%m-%d')}  ({r['days_remaining']} days remaining)")

    print()

    # Flood labels
    fl = check_flood_labels()
    session_str = "authenticated" if fl["session"] else "NOT authenticated"
    print(f"  [ ] Flood labels  (Stage 5 ground truth)")
    print(f"         Session      : {session_str}")
    print(f"         Messages     : {fl['msg_count']}")
    if fl["msg_first"]:
        print(f"         Date range   : {fl['msg_first']}  to  {fl['msg_last']}")
    if fl["last_sync"]:
        print(f"         Last sync    : {fl['last_sync']} UTC")
    if fl["type_counts"]:
        for mtype, n in sorted(fl["type_counts"].items()):
            print(f"           {mtype:<14} {n}")

    print()

    # Stages 4-5
    blocked = not r["complete"]
    for name, desc in [("Stage 4", "Nowcaster training"), ("Stage 5", "Evaluation")]:
        done = flag(name.lower().replace(" ", ""))
        mark = "+" if done else " "
        note = "  [blocked - awaiting Stage 3]" if blocked and not done else ""
        print(f"  [{mark}] {name}  {desc}{note}")
        blocked = not done  # stage 5 blocked until stage 4

    print()
    print("  Next action:", end=" ")
    if not r["complete"]:
        print(f"Wait for Stage 3 to complete (~{r['eta'].strftime('%Y-%m-%d')}).")
        print("  Run periodically: python scripts/preprocess_radar.py  (appends new PNGs to zarr)")
    elif not flag("stage4"):
        print("Start Stage 4: python train.py training.max_steps=100 training.batch_size=2")
    elif not flag("stage5"):
        print("Start Stage 5: python scripts/evaluate.py")
    else:
        print("All stages complete.")

    print()


if __name__ == "__main__":
    main()
