import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
TG_TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
RECEIPTS_FILE = "receipts.csv"
RATE_DELAY    = 1.0   # seconds between requests
MAX_RETRIES   = 3

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

USCIS_URL = "https://egov.uscis.gov/case-status/api/cases/{}"

STATUS_EMOJIS = {
    "approved":          "✅",
    "card was mailed":   "📬",
    "fingerprint":       "🔍",
    "received":          "📥",
    "rejected":          "❌",
    "denied":            "🚫",
    "withdrawn":         "↩️",
    "interview":         "🗓",
}

def get_emoji(status: str) -> str:
    s = status.lower()
    for kw, emoji in STATUS_EMOJIS.items():
        if kw in s:
            return emoji
    return "🔄"

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram] Failed to send message: {e}")

def fetch_case(receipt: str) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                USCIS_URL.format(receipt),
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if r.status_code == 429:
                wait = 30 * attempt
                print(f"[{receipt}] Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            # USCIS response shape: {"case_status": {"current_case_status_text_en": "...", "form_type": "...", ...}}
            cs = data.get("case_status", {})
            return {
                "receipt_number":  receipt,
                "form_type":       cs.get("form_type", ""),
                "status":          cs.get("current_case_status_text_en", "Unknown"),
                "action_date":     cs.get("case_status_date", None),
                "description":     cs.get("current_case_status_desc_en", ""),
                "service_center":  receipt[:3],   # IOE, WAC, EAC, LIN, SRC, NBC
                "fetched_at":      datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"[{receipt}] Attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    return None

def get_last_status(receipt: str) -> str | None:
    res = (
        sb.table("status_log")
        .select("status")
        .eq("receipt_number", receipt)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["status"]
    return None

def upsert_case(case: dict):
    sb.table("cases").upsert(
        {
            "receipt_number": case["receipt_number"],
            "form_type":      case["form_type"],
            "service_center": case["service_center"],
            "last_status":    case["status"],
            "action_date":    case["action_date"],
            "updated_at":     case["fetched_at"],
        },
        on_conflict="receipt_number"
    ).execute()

def insert_status_log(case: dict):
    sb.table("status_log").insert(
        {
            "receipt_number": case["receipt_number"],
            "status":         case["status"],
            "description":    case["description"],
            "action_date":    case["action_date"],
            "fetched_at":     case["fetched_at"],
        }
    ).execute()

def main():
    df = pd.read_csv(RECEIPTS_FILE, dtype=str)
    receipts = df["receipt_number"].str.strip().dropna().tolist()

    total     = len(receipts)
    success   = 0
    changed   = 0
    failed    = 0
    changes   = []

    print(f"[fetch_cases] Starting fetch for {total} receipts")
    send_telegram(f"🚀 <b>USCIS Fetch Started</b>\nProcessing <b>{total}</b> receipts (Form 821-D)")

    start = time.time()

    for i, receipt in enumerate(receipts, 1):
        case = fetch_case(receipt)

        if case is None:
            failed += 1
            print(f"[{i}/{total}] {receipt} — FAILED")
        else:
            prev_status = get_last_status(receipt)
            upsert_case(case)
            insert_status_log(case)
            success += 1

            emoji = get_emoji(case["status"])
            status_short = case["status"][:60]

            if prev_status is None:
                # First time seeing this receipt
                changes.append(f"{emoji} <code>{receipt}</code> — First pull: {status_short}")
                changed += 1
            elif prev_status != case["status"]:
                # Status changed — notify immediately
                changed += 1
                msg = (
                    f"🔔 <b>Status Change Detected</b>\n"
                    f"Receipt: <code>{receipt}</code>\n"
                    f"From: {prev_status}\n"
                    f"To:   {emoji} {status_short}\n"
                    f"Date: {case['action_date'] or 'N/A'}"
                )
                send_telegram(msg)
                changes.append(f"↕️ <code>{receipt}</code> changed → {status_short}")

            print(f"[{i}/{total}] {receipt} — {status_short}")

        # Respect rate limit
        time.sleep(RATE_DELAY)

    elapsed = round(time.time() - start)
    mins    = elapsed // 60
    secs    = elapsed % 60

    # Build summary
    summary_lines = [
        f"✅ <b>USCIS Fetch Complete</b>",
        f"",
        f"📋 Total:    <b>{total}</b>",
        f"✅ Success:  <b>{success}</b>",
        f"↕️ Changed:  <b>{changed}</b>",
        f"❌ Failed:   <b>{failed}</b>",
        f"⏱ Duration: <b>{mins}m {secs}s</b>",
    ]

    if changes:
        summary_lines.append("")
        summary_lines.append("<b>Changes this run:</b>")
        # Cap at 20 lines to stay under Telegram's 4096 char limit
        for line in changes[:20]:
            summary_lines.append(f"  {line}")
        if len(changes) > 20:
            summary_lines.append(f"  ... and {len(changes) - 20} more")

    send_telegram("\n".join(summary_lines))
    print(f"\n[fetch_cases] Done. {success}/{total} success, {changed} changed, {failed} failed.")

if __name__ == "__main__":
    main()
