#!/usr/bin/env python3
"""
Watch flood_labels.parquet for FLASH_FLOOD events and alert when one registers.

Runs as the last action of the daily 'SG-Weather Telegram Labels' scheduled
task (after collect -> geocode -> build). FLASH_FLOOD messages are the real
ground truth for Stage 5 flood scoring; none have occurred since collection
began 2026-05-22, so this watcher exists to make the first one impossible to
miss:

- writes checkpoints/FLASH_FLOOD_REGISTERED.flag with event details (durable,
  survives missed toasts while travelling)
- fires a Windows toast notification (best-effort)

Already-alerted events are tracked by message_id in a state file, so the
watcher only fires once per event and is safe to re-run.

Usage:
    python scripts/check_flash_flood.py           # check + alert on new events
    python scripts/check_flash_flood.py --status  # report, never alert
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "processed" / "flood_labels.parquet"
STATE_PATH = ROOT / "data" / "raw" / "flood_labels" / "flash_flood_alert_state.json"
FLAG_PATH = ROOT / "checkpoints" / "FLASH_FLOOD_REGISTERED.flag"

# Known-good AppUserModelID for toasts from an unpackaged script host.
POWERSHELL_APPID = (
    "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"
)


def load_state() -> set[int]:
    if STATE_PATH.exists():
        return set(json.loads(STATE_PATH.read_text(encoding="utf-8")).get("alerted_ids", []))
    return set()


def save_state(alerted_ids: set[int]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"alerted_ids": sorted(alerted_ids)}, indent=2), encoding="utf-8"
    )


def show_toast(title: str, message: str) -> bool:
    """Fire a Windows toast. Best-effort: returns False on any failure."""
    # Toast XML is quote-sensitive; keep only characters that cannot break it.
    safe = re.sub(r"[^A-Za-z0-9 :,.\-/()+]", " ", message)[:200]
    ps = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast scenario="urgent"><visual><binding template="ToastGeneric"><text>{title}</text><text>{safe}</text></binding></visual></toast>')
$toast = New-Object Windows.UI.Notifications.ToastNotification($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{POWERSHELL_APPID}').Show($toast)
"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Alert when a FLASH_FLOOD label registers")
    parser.add_argument("--status", action="store_true", help="Report counts, never alert")
    args = parser.parse_args()

    if not PARQUET_PATH.exists():
        print("flood_labels.parquet not found; nothing to check.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    ff = df[df["message_type"] == "FLASH_FLOOD"]
    print(f"FLASH_FLOOD labels: {len(ff)} of {len(df)} total")

    if args.status or ff.empty:
        return

    alerted = load_state()
    new = ff[~ff["message_id"].isin(alerted)]
    if new.empty:
        print("No new events (all previously alerted).")
        return

    lines = []
    for _, ev in new.iterrows():
        locs = json.loads(ev["locations"]) if isinstance(ev["locations"], str) else ev["locations"]
        loc = locs[0] if locs else "location unknown"
        lines.append(f"{ev['event_datetime']}  {loc}")
        print(f"NEW FLASH_FLOOD: {lines[-1]}")

    FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = FLAG_PATH.read_text(encoding="utf-8") if FLAG_PATH.exists() else ""
    FLAG_PATH.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")

    toast_ok = show_toast(
        "SG Weather: FLASH FLOOD registered",
        f"{len(new)} new event(s). Latest: {lines[-1]}",
    )
    print(f"Flag written: {FLAG_PATH.name} | toast: {'shown' if toast_ok else 'FAILED (flag file is the fallback)'}")

    save_state(alerted | set(new["message_id"]))


if __name__ == "__main__":
    main()
