#!/usr/bin/env python3
"""
Watch the radar archive for missing frames and alert while they are still
recoverable.

Runs as the last action of the daily 'SG-Weather Telegram Labels' scheduled
task, alongside check_flash_flood.py.

The archive is the product of this project, and its only real failure mode is
incompleteness. But a gap is not itself news — the laptop is shut for travel
routinely and the daily catch-up scraper heals those holes on its own. What
matters is whether a gap is still *fixable*: NEA serves roughly 6.5-7 days of
history (measured 2026-09-22: -156h served, -168h returned 404), and once a gap
ages past that it is lost permanently and no alert can help.

So severity is keyed to recoverability, not to existence:

  CRITICAL  recoverable but aging (>4d old, ~2.5d left to act), or overlapping
            a flood label — a gap there silently deletes ground truth from both
            training and evaluation, because RadarDataset._contiguous drops any
            sample whose context+lead span crosses a hole
  WARN      recent gap (the catch-up task should heal it); or the daily
            preprocess has not run in >26h (the zarr being merely *behind* the
            PNGs is normal between daily runs and is not reported)
  INFO      nothing to be done - recorded once, then suppressed forever so it
            cannot train the reader to ignore gap alerts. Two kinds: older than
            retention, or still in-window but never published upstream (NEA
            returns 404 for some slots it simply never served)

Recoverability of an in-window gap is VERIFIED against NEA, not assumed - see
probe_upstream(). Use --no-probe to skip that network check.

- writes checkpoints/RADAR_GAP_ALERT.flag with details (durable, survives
  missed toasts while travelling)
- fires a Windows toast notification (best-effort)

Already-alerted gaps are tracked in a state file. CRITICAL gaps re-alert at
most once every 24h while they remain actionable, because their window is
closing; everything else fires once.

Usage:
    python scripts/check_radar_gaps.py           # check + alert
    python scripts/check_radar_gaps.py --status  # report, never alert
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RADAR_DIR = ROOT / "data" / "raw" / "radar"
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
LABELS_PATH = ROOT / "data" / "processed" / "flood_labels.parquet"
STATE_PATH = ROOT / "data" / "raw" / "radar_gap_alert_state.json"
FLAG_PATH = ROOT / "checkpoints" / "RADAR_GAP_ALERT.flag"

# Known-good AppUserModelID for toasts from an unpackaged script host.
POWERSHELL_APPID = (
    "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"
)

RADAR_INTERVAL_MIN = 5
# Conservative: 156h was served, 168h was not. Treat anything past this as lost.
RETENTION_H = 156
# A gap older than this has burned most of its recovery window -> page.
AGING_THRESHOLD_H = 96
# Don't flag the freshest slots; NEA publishes with some lag and the scraper
# runs every 30 min, so the tail is expected to be briefly incomplete.
FRESH_LAG_MIN = 45
# A flood sample spans context_frames + target_offset (6 + 6 frames = +/-30 min).
FLOOD_SPAN_MIN = 30
# Re-page an actionable CRITICAL gap at most this often.
REALERT_H = 24
# The zarr is only stale if the DAILY preprocess did not run: 24h cadence + grace.
ZARR_STALE_H = 26
# Upstream probe: how many slots to sample per in-window gap, and the rate limit
# (matches scrape_radar.py's REQUEST_INTERVAL_S).
PROBE_SAMPLES = 4
PROBE_INTERVAL_S = 1.2

RADAR_URL_TEMPLATE = (
    "https://www.weather.gov.sg/files/rainarea/50km/v2/"
    "dpsri_70km_{ts}0000dBR.dpsri.png"
)
RADAR_REFERER = "https://www.weather.gov.sg/weather-rain-area-50km/"

SEV_ORDER = {"CRITICAL": 0, "WARN": 1, "INFO": 2}


# ── state ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"alerted": {}, "unrecoverable_seen": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


# ── toast (same shape as check_flash_flood.py) ────────────────────────────────

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


# ── gap detection ─────────────────────────────────────────────────────────────

def archived_timestamps() -> list[datetime]:
    out = []
    for f in RADAR_DIR.glob("*.png"):
        try:
            out.append(datetime.strptime(f.stem, "%Y%m%d_%H%M"))
        except ValueError:
            continue  # not a frame filename; ignore
    return sorted(out)


def find_gaps(present: list[datetime], now: datetime) -> list[dict]:
    """Contiguous runs of absent 5-minute slots, oldest first."""
    if not present:
        return []
    have = set(present)
    start = present[0]
    end = now - timedelta(minutes=FRESH_LAG_MIN)
    step = timedelta(minutes=RADAR_INTERVAL_MIN)

    gaps: list[dict] = []
    run_start = None
    t = start
    while t <= end:
        if t in have:
            if run_start is not None:
                gaps.append({"start": run_start, "end": t - step})
                run_start = None
        elif run_start is None:
            run_start = t
        t += step
    if run_start is not None:
        gaps.append({"start": run_start, "end": end})

    for g in gaps:
        span = g["end"] - g["start"]
        g["slots"] = int(span / step) + 1
        g["age_h"] = (now - g["end"]).total_seconds() / 3600
        g["recoverable"] = g["age_h"] < RETENTION_H
        g["id"] = g["start"].strftime("%Y%m%d_%H%M")
    return gaps


def flood_times() -> list[datetime]:
    """Naive-UTC times of flood/risk labels, for gap-overlap checking."""
    if not LABELS_PATH.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(LABELS_PATH)
    except Exception:
        return []
    if "message_type" not in df.columns or "event_datetime" not in df.columns:
        return []
    ev = df[df["message_type"].isin(["FLASH_FLOOD", "FLOOD_RISK"])]["event_datetime"]
    ts = pd.to_datetime(ev, utc=True, errors="coerce").dropna()
    return [t.tz_convert(None).to_pydatetime() for t in ts]


def probe_upstream(g: dict, session) -> bool:
    """Is any frame in this gap actually still served by NEA?

    Being inside the retention window is not the same as being available: NEA
    simply never published some slots (confirmed 2026-09-22 — four in-window
    frames all returned 404). Without this check those gaps page as CRITICAL
    every 24h forever and nothing can ever clear them, which is the exact
    alert-fatigue failure this design is meant to avoid. So recoverability is
    verified, not assumed.

    Samples up to PROBE_SAMPLES slots spread across the gap; one success is
    enough, because the scraper will then fetch whatever exists.
    """
    step = timedelta(minutes=RADAR_INTERVAL_MIN)
    slots = [g["start"] + step * i for i in range(g["slots"])]
    if len(slots) > PROBE_SAMPLES:
        stride = len(slots) / PROBE_SAMPLES
        slots = [slots[int(i * stride)] for i in range(PROBE_SAMPLES)]

    for dt in slots:
        ts = (dt + timedelta(hours=8)).strftime("%Y%m%d%H%M")  # NEA serves SGT
        try:
            r = session.get(RADAR_URL_TEMPLATE.format(ts=ts), timeout=15)
            if r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n":
                return True
        except Exception:
            return True  # network trouble: don't silently downgrade a real gap
        time.sleep(PROBE_INTERVAL_S)
    return False


def classify(gaps: list[dict], floods: list[datetime], probe: bool = True) -> None:
    """Attach severity + reason to each gap, per the alert table in the plan."""
    pad = timedelta(minutes=FLOOD_SPAN_MIN)

    session = None
    if probe and any(g["recoverable"] for g in gaps):
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": RADAR_REFERER})

    for g in gaps:
        hit = [f for f in floods if g["start"] - pad <= f <= g["end"] + pad]
        g["floods"] = hit
        g["upstream"] = None
        if g["recoverable"] and session is not None:
            g["upstream"] = probe_upstream(g, session)

        if not g["recoverable"]:
            # Nothing can be done; record once and never page again.
            g["severity"], g["reason"] = "INFO", "older than retention - unrecoverable"
        elif g["upstream"] is False:
            # In-window but NEA has nothing to give. Not actionable, so not a page.
            g["severity"], g["reason"] = "INFO", "upstream 404 - never published"
        elif hit:
            g["severity"] = "CRITICAL"
            g["reason"] = f"overlaps {len(hit)} flood label(s) - ground truth at risk"
        elif g["age_h"] > AGING_THRESHOLD_H:
            left = RETENTION_H - g["age_h"]
            g["severity"] = "CRITICAL"
            g["reason"] = f"recoverable for only ~{left / 24:.1f}d more"
        else:
            g["severity"], g["reason"] = "WARN", "recent - catch-up scraper should heal"


def runbook(g: dict) -> str:
    hours = int(g["age_h"]) + 6
    return (
        f"python scripts/scrape_radar.py --hours {hours}"
        "  ->  python scripts/preprocess_radar.py"
        "  ->  python scripts/build_flood_eval_dataset.py"
    )


# ── the other questions (Q4, Q5) ──────────────────────────────────────────────

def zarr_lag(n_png: int, now: datetime) -> dict | None:
    """Q4: is the zarr behind the PNGs? Eval joins read the zarr, not the PNGs."""
    if not ZARR_PATH.exists():
        return None
    try:
        import xarray as xr
        import pandas as pd
        ds = xr.open_zarr(ZARR_PATH)
        n = int(ds.sizes["time"])
        last = pd.to_datetime(ds.time.values[-1]).to_pydatetime()
        ds.close()
    except Exception as e:
        return {"error": str(e)[:80]}
    behind = n_png - n
    lag_h = (now - last).total_seconds() / 3600
    # "behind by any frames" is NOT staleness: preprocess_radar.py runs once a
    # day (2nd action of '\SG-Weather\SG-Weather Radar Scraper', 08:00 SGT) while
    # the scraper collects every 30 min, so the zarr is *normally* up to a day
    # behind. This check runs at 09:00, one hour after preprocess, where it would
    # see ~12 frames behind and warn every single day forever.
    # Only a preprocess that did not happen is worth reporting.
    return {
        "frames": n, "behind": behind, "last": last, "lag_h": lag_h,
        "stale": lag_h > ZARR_STALE_H,
    }


def task_health() -> list[tuple[str, str, str]]:
    """Q5: a dead scheduled task looks exactly like clear weather."""
    try:
        r = subprocess.run(
            ["schtasks", "/Query", "/FO", "CSV", "/V"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []
    import csv, io
    rows = []
    try:
        rdr = csv.DictReader(io.StringIO(r.stdout))
        for row in rdr:
            name = (row.get("TaskName") or "")
            if "SG-Weather" not in name:
                continue
            rows.append((name.split("\\")[-1],
                         row.get("Last Result", "?"),
                         row.get("Last Run Time", "?")))
    except csv.Error:
        return []
    return rows


# ── reporting ─────────────────────────────────────────────────────────────────

def report(gaps: list[dict], zl: dict | None, tasks, n_png: int) -> None:
    print(f"Radar archive: {n_png:,} PNGs | retention {RETENTION_H}h "
          f"({RETENTION_H / 24:.1f}d) | fresh-lag {FRESH_LAG_MIN}min")
    if not gaps:
        print("No gaps. Archive is complete.")
    else:
        # Only actionable gaps get a row. Unrecoverable ones are archaeology:
        # there are hundreds from the early archive and listing them buries the
        # two that still matter.
        live = [g for g in gaps if g["severity"] != "INFO"]
        lost = [g for g in gaps if g["severity"] == "INFO"]

        if live:
            print(f"{len(live)} actionable gap(s):\n")
            print(f"  {'SEV':<9}{'gap start (UTC)':<18}{'slots':>6}{'age':>8}  flood  reason")
            for g in sorted(live, key=lambda x: (SEV_ORDER[x["severity"]], -x["age_h"])):
                print(f"  {g['severity']:<9}{g['start']:%Y-%m-%d %H:%M}  {g['slots']:>6}"
                      f"{g['age_h']:>7.1f}h  {'YES' if g['floods'] else '-':<6} {g['reason']}")
        else:
            print("No actionable gaps.")

        if lost:
            # Two distinct dead ends, and conflating them hides a real fact:
            # one is "too late", the other is "NEA never had it".
            expired = [g for g in lost if not g["recoverable"]]
            absent = [g for g in lost if g["recoverable"]]
            if expired:
                slots = sum(g["slots"] for g in expired)
                print(f"\n  {len(expired)} gap(s) past retention "
                      f"({slots:,} frames, {min(g['start'] for g in expired):%Y-%m-%d} to "
                      f"{max(g['end'] for g in expired):%Y-%m-%d}) - suppressed, too late")
            if absent:
                slots = sum(g["slots"] for g in absent)
                print(f"  {len(absent)} gap(s) in-window but never published upstream "
                      f"({slots} frames) - suppressed, NEA has no data to serve:")
                for g in sorted(absent, key=lambda x: x["start"]):
                    print(f"      {g['start']:%Y-%m-%d %H:%M} ({g['slots']} slots)")
            with_flood = [g for g in lost if g["floods"]]
            if with_flood:
                print(f"    of which {len(with_flood)} overlapped a flood label "
                      f"(permanently lost ground truth):")
                for g in with_flood:
                    print(f"      {g['start']:%Y-%m-%d %H:%M} ({g['slots']} slots)")

        actionable = [g for g in gaps if g["severity"] == "CRITICAL"]
        if actionable:
            print("\nRunbook (oldest actionable gap first):")
            for g in sorted(actionable, key=lambda x: -x["age_h"])[:3]:
                print(f"  {g['id']}: {runbook(g)}")

    print()
    if zl is None:
        print("zarr: not found")
    elif "error" in zl:
        print(f"zarr: unreadable ({zl['error']})")
    else:
        flag = f"STALE - daily preprocess has not run in {zl['lag_h']:.0f}h"             if zl["stale"] else "ok (daily preprocess cadence)"
        print(f"zarr: {zl['frames']:,} frames, last {zl['last']:%Y-%m-%d %H:%M} "
              f"({zl['lag_h']:.1f}h ago), {zl['behind']:+d} vs PNGs -> {flag}")
    if tasks:
        print("scheduled tasks:")
        for name, res, last in tasks:
            print(f"  {'ok ' if res == '0' else 'BAD'} {name:<34} result={res:<5} last={last}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Alert on recoverable radar archive gaps")
    parser.add_argument("--status", action="store_true", help="Report only, never alert")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip the upstream availability check (offline/fast)")
    args = parser.parse_args()

    if not RADAR_DIR.exists():
        print("Radar archive directory not found; nothing to check.")
        return

    now = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    present = archived_timestamps()
    if not present:
        print("Radar archive is empty; nothing to check.")
        return

    gaps = find_gaps(present, now)
    classify(gaps, flood_times(), probe=not args.no_probe)
    zl = zarr_lag(len(present), now)
    tasks = task_health()

    report(gaps, zl, tasks, len(present))

    if args.status:
        return

    state = load_state()
    alerted: dict = state.get("alerted", {})
    seen_lost: set = set(state.get("unrecoverable_seen", []))

    # INFO gaps: record once, then suppressed forever. This is what keeps the
    # permanently-lost May/June holes from paging on every single run.
    for g in gaps:
        if g["severity"] == "INFO":
            seen_lost.add(g["id"])

    fire = []
    for g in gaps:
        if g["severity"] == "INFO":
            continue
        if g["severity"] == "WARN":
            if g["id"] not in alerted:
                fire.append(g)
            continue
        last = alerted.get(g["id"])
        if last is None:
            fire.append(g)
        else:
            try:
                age = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            except ValueError:
                age = REALERT_H + 1
            if age >= REALERT_H:  # window is closing; nudge again
                fire.append(g)

    stale_zarr = bool(zl and zl.get("stale"))
    bad_tasks = [t for t in tasks if t[1] not in ("0", "?")]

    if not fire and not stale_zarr and not bad_tasks:
        print("\nNo new alerts.")
        state["alerted"] = alerted
        state["unrecoverable_seen"] = sorted(seen_lost)
        save_state(state)
        return

    lines = [f"# {now:%Y-%m-%d %H:%M} UTC"]
    for g in sorted(fire, key=lambda x: (SEV_ORDER[x["severity"]], -x["age_h"])):
        lines.append(f"{g['severity']}  gap {g['start']:%Y-%m-%d %H:%M} -> "
                     f"{g['end']:%Y-%m-%d %H:%M}  ({g['slots']} slots, "
                     f"{g['age_h']:.1f}h old) - {g['reason']}")
        if g["severity"] == "CRITICAL":
            lines.append(f"    fix: {runbook(g)}")
        alerted[g["id"]] = now.isoformat()
    if stale_zarr:
        lines.append(f"WARN  daily preprocess has not run in {zl['lag_h']:.0f}h - zarr "
                     f"{zl['behind']} frames behind PNGs (last {zl['last']:%Y-%m-%d %H:%M}) - "
                     f"fix: python scripts/preprocess_radar.py")
    for name, res, last in bad_tasks:
        lines.append(f"WARN  scheduled task '{name}' last result {res} (last run {last})")

    FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = FLAG_PATH.read_text(encoding="utf-8") if FLAG_PATH.exists() else ""
    FLAG_PATH.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")

    print("\nNEW ALERTS:")
    for ln in lines[1:]:
        print(" ", ln)

    crit = [g for g in fire if g["severity"] == "CRITICAL"]
    if crit or bad_tasks:
        worst = min(crit, key=lambda g: RETENTION_H - g["age_h"]) if crit else None
        msg = (f"{len(crit)} recoverable gap(s) aging out. Oldest: "
               f"{worst['start']:%d %b %H:%M}, {worst['slots']} frames."
               if worst else "Scheduled task failure - see flag file.")
        toast_ok = show_toast("SG Weather: radar archive gap", msg)
    else:
        toast_ok = show_toast(
            "SG Weather: radar archive gap",
            f"{len(fire)} new gap(s) logged; catch-up scraper should heal.",
        )
    print(f"Flag written: {FLAG_PATH.name} | "
          f"toast: {'shown' if toast_ok else 'FAILED (flag file is the fallback)'}")

    state["alerted"] = alerted
    state["unrecoverable_seen"] = sorted(seen_lost)
    save_state(state)


if __name__ == "__main__":
    main()
