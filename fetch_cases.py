import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TG_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

# ── Receipt range config ───────────────────────────────────────────────────
# Format: IOE + 2-digit year + 2-digit week + 6-digit sequence
# Example: IOE2490100001 = year 24, week 90 (USCIS internal), seq 000001
# Adjust these two values to define your search window.
IOE_PREFIX = "IOE"
IOE_START  = int(os.environ.get("IOE_START", "2490100001"))  # 10-digit suffix
IOE_END    = int(os.environ.get("IOE_END",   "2490199999"))  # inclusive

RATE_DELAY  = 1.0  # seconds between requests — do not go below 0.5
MAX_RETRIES = 3

USCIS_URL   = "https://egov.uscis.gov/case-status/api/cases/{}"

# ── DACA-specific status taxonomy ─────────────────────────────────────────
# Each status maps to: (emoji, alert_tier, pipeline_score)
#   alert_tier 1 = immediate individual alert
#   alert_tier 2 = included in run summary only
#   pipeline_score used by predict.py for ML features
DACA_STATUSES = {
    # ── Positive / forward movement ────────────────────────────────────────
    "case was approved":                          ("✅", 1, 9),
    "card is being produced":                     ("🖨️",  1, 8),
    "card was mailed":                            ("📬", 1, 9),
    "card was delivered":                         ("🏠", 1, 9),
    "employment authorization document approved": ("✅", 1, 9),
    "renewal approved":                           ("✅", 1, 9),
    "biometrics appointment was scheduled":       ("🗓️", 1, 4),
    "biometrics were taken":                      ("🔍", 2, 5),
    "fingerprints were taken":                    ("🔍", 2, 5),
    "case is being actively reviewed":            ("⚙️",  2, 6),
    "interview was scheduled":                    ("📅", 1, 7),

    # ── Needs attention ────────────────────────────────────────────────────
    "request for evidence":                       ("⚠️", 1, 3),
    "rfe":                                        ("⚠️", 1, 3),
    "response to request for evidence received":  ("📨", 1, 4),
    "notice of intent to deny":                   ("🚨", 1, 2),
    "noid":                                       ("🚨", 1, 2),

    # ── Negative outcomes ──────────────────────────────────────────────────
    "case was denied":                            ("❌", 1, 0),
    "rejected":                                   ("🚫", 1, 0),
    "administratively closed":                    ("🗂️",  1, 1),
    "terminated":                                 ("🛑", 1, 0),
    "withdrawn":                                  ("↩️", 2, 0),

    # ── Neutral / early pipeline ───────────────────────────────────────────
    "case was received":                          ("📥", 2, 1),
    "case was updated":                           ("🔄", 2, 2),
    "name was updated":                           ("✏️",  2, 2),
    "fees were waived":                           ("💸", 2, 2),
    "fees were received":                         ("💰", 2, 2),
}

APPROVAL_KEYWORDS = [
    "approved", "card was mailed", "card is being produced",
    "card was delivered", "employment authorization document approved",
    "renewal approved",
]

def match_status(raw_status: str) -> tuple:
    """Return (emoji, alert_tier, pipeline_score) for a raw USCIS status string."""
    s = raw_status.lower()
    for kw, vals in DACA_STATUSES.items():
        if kw in s:
            return vals
    return ("🔄", 2, 2)  # default: neutral update, include in summary

def is_daca_case(data: dict) -> bool:
    """Return True only if the API response is for Form I-821D."""
    cs = data.get("case_status", {})
    form = cs.get("form_type", "").upper()
    return "821" in form or "I821" in form or form == ""  # "" = form unknown, still process

def is_approved(status: str) -> bool:
    s = status.lower()
    return any(kw in s for kw in APPROVAL_KEYWORDS)

# ── Telegram ───────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram] {e}")

# ── USCIS fetch ────────────────────────────────────────────────────────────
def fetch_case(receipt: str) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                USCIS_URL.format(receipt),
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if r.status_code == 429:
                wait = 30 * attempt
                print(f"  [{receipt}] Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None  # receipt doesn't exist, skip silently
            r.raise_for_status()
            data = r.json()

            if not is_daca_case(data):
                print(f"  [{receipt}] Skipped — not an I-821D case")
                return None

            cs = data.get("case_status", {})
            raw_status = cs.get("current_case_status_text_en", "Unknown")
            emoji, alert_tier, score = match_status(raw_status)

            return {
                "receipt_number": receipt,
                "form_type":      cs.get("form_type", "I-821D"),
                "status":         raw_status,
                "status_emoji":   emoji,
                "alert_tier":     alert_tier,
                "pipeline_score": score,
                "action_date":    cs.get("case_status_date", None),
                "description":    cs.get("current_case_status_desc_en", ""),
                "service_center": "IOE",
                "fetched_at":     datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"  [{receipt}] Attempt {attempt} error: {e}")
            time.sleep(5 * attempt)
    return None

# ── Supabase helpers ───────────────────────────────────────────────────────
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_last_status(receipt: str) -> str | None:
    res = (
        sb.table("status_log")
        .select("status")
        .eq("receipt_number", receipt)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0]["status"] if res.data else None

def upsert_case(c: dict):
    sb.table("cases").upsert(
        {
            "receipt_number": c["receipt_number"],
            "form_type":      c["form_type"],
            "service_center": c["service_center"],
            "last_status":    c["status"],
            "pipeline_score": c["pipeline_score"],
            "action_date":    c["action_date"],
            "updated_at":     c["fetched_at"],
        },
        on_conflict="receipt_number",
    ).execute()

def insert_log(c: dict):
    sb.table("status_log").insert(
        {
            "receipt_number": c["receipt_number"],
            "status":         c["status"],
            "pipeline_score": c["pipeline_score"],
            "description":    c["description"],
            "action_date":    c["action_date"],
            "fetched_at":     c["fetched_at"],
        }
    ).execute()

# ── Range generator ────────────────────────────────────────────────────────
def generate_receipts(prefix: str, start: int, end: int) -> list[str]:
    """Generate IOE receipt numbers for the given numeric suffix range."""
    return [f"{prefix}{str(n).zfill(10)}" for n in range(start, end + 1)]

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    receipts = generate_receipts(IOE_PREFIX, IOE_START, IOE_END)

    # Also merge any manual receipts.csv if it exists (optional override)
    try:
        df_manual = pd.read_csv("receipts.csv", dtype=str)
        extras = df_manual["receipt_number"].str.strip().dropna().tolist()
        # Deduplicate
        receipts = list(dict.fromkeys(receipts + extras))
        print(f"[fetch_cases] Merged {len(extras)} manual receipts from receipts.csv")
    except FileNotFoundError:
        pass

    total   = len(receipts)
    success = 0
    skipped = 0
    failed  = 0
    changed = 0
    changes = []  # tier-1 alert summaries for final report

    print(f"[fetch_cases] Scanning {total} IOE receipts ({IOE_PREFIX}{IOE_START}–{IOE_PREFIX}{IOE_END})")
    send_telegram(
        f"🚀 <b>USCIS 821-D Fetch Started</b>\n"
        f"📋 Scanning <b>{total:,}</b> IOE receipts\n"
        f"🔢 Range: <code>{IOE_PREFIX}{IOE_START}</code> → <code>{IOE_PREFIX}{IOE_END}</code>"
    )

    start_time = time.time()

    for i, receipt in enumerate(receipts, 1):
        case = fetch_case(receipt)

        if case is None:
            skipped += 1
            if i % 500 == 0:
                print(f"  [{i}/{total}] ... (last skipped/failed)")
        else:
            prev_status = get_last_status(receipt)
            upsert_case(case)
            insert_log(case)
            success += 1

            emoji       = case["status_emoji"]
            status_short = case["status"][:70]
            tier        = case["alert_tier"]

            if prev_status is None:
                # First time we've seen this receipt with a real status
                print(f"  [{i}/{total}] {receipt} NEW → {status_short}")
                if tier == 1:
                    changes.append(f"{emoji} <code>{receipt}</code> — NEW: {status_short}")
                    changed += 1

            elif prev_status != case["status"]:
                changed += 1
                _, new_tier, _ = match_status(case["status"])

                if new_tier == 1:
                    # Fire immediate individual alert
                    send_telegram(
                        f"🔔 <b>DACA Status Change</b>\n"
                        f"Receipt: <code>{receipt}</code>\n"
                        f"From: {prev_status}\n"
                        f"To:   {emoji} {status_short}\n"
                        f"Date: {case['action_date'] or 'N/A'}"
                    )
                changes.append(f"↕️ <code>{receipt}</code> → {emoji} {status_short}")
                print(f"  [{i}/{total}] {receipt} CHANGED → {status_short}")
            else:
                print(f"  [{i}/{total}] {receipt} — no change")

        # Progress ping every 500 receipts
        if i % 500 == 0:
            elapsed = int(time.time() - start_time)
            send_telegram(
                f"⏳ <b>Progress update</b>\n"
                f"{i:,} / {total:,} receipts scanned\n"
                f"✅ Found: {success} · ↕️ Changed: {changed} · ⏭ Skipped: {skipped}\n"
                f"⏱ Elapsed: {elapsed // 60}m {elapsed % 60}s"
            )

        time.sleep(RATE_DELAY)

    elapsed = int(time.time() - start_time)
    mins, secs = elapsed // 60, elapsed % 60

    summary = [
        f"✅ <b>USCIS 821-D Fetch Complete</b>",
        f"",
        f"📋 Scanned:  <b>{total:,}</b>",
        f"✅ Found:    <b>{success:,}</b> active I-821D cases",
        f"↕️ Changed:  <b>{changed}</b>",
        f"⏭ Skipped:  <b>{skipped:,}</b> (404 / non-DACA)",
        f"⏱ Duration: <b>{mins}m {secs}s</b>",
    ]
    if changes:
        summary += ["", "<b>Notable changes this run:</b>"]
        for line in changes[:25]:
            summary.append(f"  {line}")
        if len(changes) > 25:
            summary.append(f"  ... and {len(changes) - 25} more")

    send_telegram("\n".join(summary))
    print(f"\n[fetch_cases] Done. {success}/{total} found, {changed} changed, {skipped} skipped.")

if __name__ == "__main__":
    main()
