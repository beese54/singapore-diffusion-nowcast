#!/usr/bin/env python3
"""
Collect and parse PUB Telegram flood alerts for Stage 5 evaluation ground truth.

Fetches @pubfloodalerts messages from 2026-05-22 onwards (incremental),
parses them with a rule-based classifier, and writes flood_labels.parquet.

Usage:
    python scripts/collect_flood_labels.py           # fetch new + update parquet
    python scripts/collect_flood_labels.py --status  # report counts, no fetch

First run: you will be prompted for your Telegram phone number + SMS code.
Subsequent runs (including Task Scheduler) are headless once the session is saved.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from telethon import TelegramClient
except ImportError:
    TelegramClient = None  # deferred error shown only when fetch is attempted

load_dotenv()

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw" / "flood_labels"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINTS_DIR = ROOT / "checkpoints"

# Telethon stores session as <path>.session
SESSION_PATH = str(CHECKPOINTS_DIR / "telegram")
STATE_PATH = RAW_DIR / "state.json"
MESSAGES_PATH = RAW_DIR / "messages.json"
PARQUET_PATH = PROCESSED_DIR / "flood_labels.parquet"

# Only collect messages on/after this date — matches radar archive start
CUTOFF_DATE = datetime(2026, 5, 22, 0, 0, 0, tzinfo=timezone.utc)
PUB_CHANNEL = "pubfloodalerts"

FLOOD_KEYWORDS = [
    "[flash flood occurred]",
    "[risk of flash floods]",
    "flash flood",
    "flooding",
    "flood",
    "water level",
    "water rises",
    "inundated",
    "submerged",
    "canal overflow",
    "heavy rain",
]


# ---------------------------------------------------------------------------
# Rule-based parser (mirrors predict_flash_flood/src/parse/telegram_message_parser.py)
# ---------------------------------------------------------------------------

def _classify(text: str) -> str:
    lower = text.lower()
    if "[flash flood occurred]" in lower:
        return "FLASH_FLOOD"
    if "[risk of flash floods]" in lower:
        return "FLOOD_RISK"
    if "heavy rain" in lower:
        return "RAIN_WARNING"
    return "OTHER"


def _extract_locations(text: str, msg_type: str) -> list[str]:
    if msg_type == "FLASH_FLOOD":
        m = re.search(r"[Ff]lash flood at (.+?)\.", text)
        return [m.group(1).strip()] if m else []
    if msg_type == "FLOOD_RISK":
        m = re.search(r"avoid.*?:\s*(.+)", text, re.IGNORECASE)
        if m:
            loc = m.group(1).strip()
            loc = re.sub(r"\s+for the next.+", "", loc, flags=re.IGNORECASE).strip()
            return [loc] if loc else []
    return []


def _extract_event_time(text: str, fallback: datetime) -> datetime:
    m = re.search(r"(\d{1,2}:\d{2})\s*hours", text, re.IGNORECASE)
    if m:
        try:
            t = datetime.strptime(m.group(1), "%H:%M")
            return fallback.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        except ValueError:
            pass
    m = re.search(r"\b(\d{4})\s*hours\b", text, re.IGNORECASE)
    if m:
        try:
            t = datetime.strptime(m.group(1), "%H%M")
            return fallback.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        except ValueError:
            pass
    return fallback


def parse_message(msg: dict) -> dict:
    text = msg.get("text") or ""
    msg_type = _classify(text)
    fallback_dt = datetime.fromisoformat(msg["date"])
    if fallback_dt.tzinfo is None:
        fallback_dt = fallback_dt.replace(tzinfo=timezone.utc)
    event_dt = _extract_event_time(text, fallback_dt)
    locations = _extract_locations(text, msg_type)
    return {
        "message_id": msg["message_id"],
        "event_datetime": event_dt.isoformat(),
        "message_type": msg_type,
        "raw_text": text,
        "locations": json.dumps(locations),
    }


# ---------------------------------------------------------------------------
# State / IO helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"last_message_id": 0, "last_run": None}


def save_state(last_id: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"last_message_id": last_id, "last_run": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def load_messages() -> list[dict]:
    if MESSAGES_PATH.exists():
        return json.loads(MESSAGES_PATH.read_text(encoding="utf-8"))
    return []


def append_messages(new_msgs: list[dict]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_messages()
    existing.extend(new_msgs)
    MESSAGES_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def rebuild_parquet(messages: list[dict]) -> int:
    if not messages:
        return 0
    rows = [parse_message(m) for m in messages]
    df = pd.DataFrame(rows)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True)
    df = df.sort_values("event_datetime").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def print_status() -> None:
    session_exists = Path(SESSION_PATH + ".session").exists()
    messages = load_messages()
    state = load_state()

    print(f"Session file  : {'found' if session_exists else 'NOT FOUND — run script interactively first to authenticate'}")
    print(f"Raw messages  : {len(messages)}")
    if messages:
        dates = [m["date"] for m in messages]
        print(f"  Date range  : {min(dates)[:10]}  to  {max(dates)[:10]}")
    last_run_str = state["last_run"][:19] + " UTC" if state["last_run"] else "never"
    print(f"Last sync     : {last_run_str}  (last_id={state['last_message_id']})")
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        print(f"Parquet rows  : {len(df)}")
        for mtype, n in df["message_type"].value_counts().items():
            print(f"  {mtype:<14} {n}")
    else:
        print("Parquet       : not yet written")


# ---------------------------------------------------------------------------
# Telegram fetch
# ---------------------------------------------------------------------------

async def _fetch(api_id: int, api_hash: str, last_id: int) -> list[dict]:
    new_msgs: list[dict] = []
    async with TelegramClient(SESSION_PATH, api_id, api_hash) as client:
        if last_id > 0:
            # Incremental: only messages with id > last_id
            iterator = client.iter_messages(PUB_CHANNEL, min_id=last_id, reverse=True, limit=None)
        else:
            # Bootstrap: start from CUTOFF_DATE, filter anything older client-side
            iterator = client.iter_messages(PUB_CHANNEL, offset_date=CUTOFF_DATE, reverse=True, limit=None)

        async for msg in iterator:
            if not msg.text:
                continue
            if msg.date < CUTOFF_DATE:
                continue
            lower = msg.text.lower()
            if not any(kw in lower for kw in FLOOD_KEYWORDS):
                continue
            new_msgs.append({
                "message_id": msg.id,
                "date": msg.date.isoformat(),
                "text": msg.text,
            })

    return new_msgs


def fetch_new_messages(api_id: int, api_hash: str, last_id: int) -> list[dict]:
    return asyncio.run(_fetch(api_id, api_hash, last_id))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_credentials() -> tuple[int, str]:
    api_id_str = os.environ.get("TELEGRAM_API_ID", "")
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    if not api_id_str or not api_hash or api_id_str.startswith("your_"):
        print("ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("       Get them at https://my.telegram.org/apps")
        sys.exit(1)
    return int(api_id_str), api_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PUB Telegram flood labels")
    parser.add_argument("--status", action="store_true", help="Report status without fetching")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if TelegramClient is None:
        print("ERROR: telethon is not installed. Run: pip install telethon")
        sys.exit(1)

    api_id, api_hash = get_credentials()

    session_file = Path(SESSION_PATH + ".session")
    if not session_file.exists():
        print("No Telegram session found. You will be prompted for your phone number and SMS code.")
        print(f"Session will be saved to: {session_file}")
        print()

    state = load_state()
    last_id = state["last_message_id"]
    print(f"Fetching messages after id={last_id} (cutoff: {CUTOFF_DATE.date()}) ...")

    new_msgs = fetch_new_messages(api_id, api_hash, last_id)

    if not new_msgs:
        print("No new messages.")
        save_state(last_id)
    else:
        append_messages(new_msgs)
        max_id = max(m["message_id"] for m in new_msgs)
        save_state(max_id)
        print(f"Saved {len(new_msgs)} new messages  (last_id now {max_id})")

    all_messages = load_messages()
    n = rebuild_parquet(all_messages)
    print(f"Parquet updated: {n} rows -> {PARQUET_PATH.relative_to(ROOT)}")

    if all_messages:
        types = [parse_message(m)["message_type"] for m in all_messages]
        for mtype, count in sorted(Counter(types).items()):
            print(f"  {mtype:<14} {count}")


if __name__ == "__main__":
    main()
