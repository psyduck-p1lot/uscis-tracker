import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

TG_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
DATABASE_URL = os.environ["DATABASE_URL"]   # postgresql://postgres:PASSWORD@db.xxx.supabase.co:5432/postgres

IOE_PREFIX = "IOE"
IOE_START  = int(os.environ.get("IOE_START", "2490100001"))
IOE_END    = int(os.environ.get("IOE_END",   "2490100050"))

RATE_DELAY  = 1.0
MAX_RETRIES = 3
USCIS_URL   = "https://egov.uscis.gov/case-status/api/cases/{}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ── DACA status taxonomy ───────────────────────────────────────────────────
DACA_STATUSES = {
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
    "request for evidence":                       ("⚠️", 1, 3),
    "response to request for evidence received":  ("📨", 1, 4),
    "notice of intent to deny":                   ("🚨", 1, 2),
    "case was denied":                            ("❌", 1, 0),
    "rejected":                                   ("🚫", 1, 0),
    "administratively closed":                    ("🗂️",  1, 1),
    "terminated":                                 ("🛑", 1, 0),
    "withdrawn":                                  ("↩️", 2, 0),
    "case was received":                          ("📥", 2, 1),
    "case was updated":                           ("🔄", 2, 2),
    "fees were received":                         ("💰", 2, 2),
}

APPROVAL_KW = [
    "approved", "card was mailed", "card is being produced",
    "card was delivered", "employment authorization document approved", "renewal approved",
]

def match_status(raw: str) -> tuple:
    s = raw.lower()
    for kw, vals in DACA_STATUSES.items():
        if kw in s:
            return vals
    return ("🔄", 2, 2)

def is_daca(data: dict) -> bool:
    form = data.get("case_status", {}).get("form_type", "").upper()
    return "821" in form or form == ""

# ── DB helpers ─────────────────────────────────────────────────────────────
def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cases (
                receipt_number  TEXT PRIMARY KEY,
                form_type       TEXT DEFAULT 'I-821D',
                service_center  TEXT DEFAULT 'IOE',
                last_status     TEXT,
                pipeline_score  INTEGER DEFAULT 0,
                action_date     TEXT,
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS status_log (
                id              BIGSERIAL PRIMARY KEY,
                receipt_number  TEXT NOT NULL,
                status          TEXT,
                pipeline_score  INTEGER DEFAULT 0,
                description     TEXT,
                action_date     TEXT,
                fetched_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                receipt_number      TEXT PRIMARY KEY,
                est_days_remaining  INTEGER,
                est_total_days      INTEGER,
                model_mae_days      FLOAT,
                confidence          FLOAT,
                had_rfe             BOOLEAN DEFAULT FALSE,
                is_renewal          BOOLEAN DEFAULT FALSE,
                predicted_at        TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_log_receipt
                ON status_log (receipt_number, fetched_at DESC)
        """))
    print("[db] Schema ready.")

def get_last_status(conn, receipt: str) -> str | None:
    row = conn.execute(
        text("SELECT status FROM status_log WHERE receipt_number=:r ORDER BY fetched_at DESC LIMIT 1"),
        {"r": receipt}
    ).fetchone()
    return row[0] if row else None

def upsert_case(conn, c: dict):
    conn.execute(text("""
        INSERT INTO cases (receipt_number, form_type, service_center, last_status,
                           pipeline_score, action_date, updated_at)
        VALUES (:receipt, :form, :sc, :status, :score, :action_date, :ts)
        ON CONFLICT (receipt_number) DO UPDATE SET
            last_status    = EXCLUDED.last_status,
            pipeline_score = EXCLUDED.pipeline_score,
            action_date    = EXCLUDED.action_date,
            updated_at     = EXCLUDED.updated_at
    """), {
        "receipt": c["receipt_number"], "form": c["form_type"],
        "sc": c["service_center"],      "status": c["status"],
        "score": c["pipeline_score"],   "action_date": c["action_date"],
        "ts": c["fetched_at"],
    })

def insert_log(conn, c: dict):
    conn.execute(text("""
        INSERT INTO status_log (receipt_number, status, pipeline_score,
                                description, action_date, fetched_at)
        VALUES (:receipt, :status, :score, :desc, :action_date, :ts)
    """), {
        "receipt": c["receipt_number"], "status": c["status"],
        "score": c["pipeline_score"],   "desc": c["description"],
        "action_date": c["action_date"], "ts": c["fetched_at"],
    })

# ── Telegram ───────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg,
                                     "parse_mode": "HTML"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram] {e}")

# ── USCIS fetch ────────────────────────────────────────────────────────────
def fetch_case(receipt: str) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(USCIS_URL.format(receipt),
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 429:
                time.sleep(30 * attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            if not is_daca(data):
                return None
            cs    = data.get("case_status", {})
            raw   = cs.get("current_case_status_text_en", "Unknown")
            emoji, tier, score = match_status(raw)
            return {
                "receipt_number": receipt,
                "form_type":      cs.get("form_type", "I-821D"),
                "status":         raw,
                "status_emoji":   emoji,
                "alert_tier":     tier,
                "pipeline_score": score,
                "action_date":    cs.get("case_status_date"),
                "description":    cs.get("current_case_status_desc_en", ""),
                "service_center": "IOE",
                "fetched_at":     datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"  [{receipt}] attempt {attempt}: {e}")
            time.sleep(5 * attempt)
    return None

def generate_receipts(prefix, start, end):
    return [f"{prefix}{str(n).zfill(10)}" for n in range(start, end + 1)]

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    init_db()

    receipts = generate_receipts(IOE_PREFIX, IOE_START, IOE_END)
    try:
        extras = pd.read_csv("receipts.csv", dtype=str)["receipt_number"].dropna().tolist()
        receipts = list(dict.fromkeys(receipts + extras))
        print(f"[fetch] Merged {len(extras)} from receipts.csv")
    except FileNotFoundError:
        pass

    total   = len(receipts)
    success = skipped = failed = changed = 0
    changes = []

    print(f"[fetch] Scanning {total} receipts: {IOE_PREFIX}{IOE_START} → {IOE_PREFIX}{IOE_END}")
    send_telegram(
        f"🚀 <b>USCIS 821-D Fetch Started</b>\n"
        f"📋 Scanning <b>{total:,}</b> IOE receipts\n"
        f"🔢 Range: <code>{IOE_PREFIX}{IOE_START}</code> → <code>{IOE_PREFIX}{IOE_END}</code>"
    )

    start_time = time.time()

    with engine.begin() as conn:
        for i, receipt in enumerate(receipts, 1):
            case = fetch_case(receipt)

            if case is None:
                skipped += 1
            else:
                prev = get_last_status(conn, receipt)
                upsert_case(conn, case)
                insert_log(conn, case)
                success += 1

                emoji = case["status_emoji"]
                short = case["status"][:70]

                if prev is None:
                    if case["alert_tier"] == 1:
                        changes.append(f"{emoji} <code>{receipt}</code> — NEW: {short}")
                        changed += 1
                elif prev != case["status"]:
                    changed += 1
                    if case["alert_tier"] == 1:
                        send_telegram(
                            f"🔔 <b>DACA Status Change</b>\n"
                            f"Receipt: <code>{receipt}</code>\n"
                            f"From: {prev}\n"
                            f"To:   {emoji} {short}\n"
                            f"Date: {case['action_date'] or 'N/A'}"
                        )
                    changes.append(f"↕️ <code>{receipt}</code> → {emoji} {short}")

            if i % 500 == 0:
                elapsed = int(time.time() - start_time)
                send_telegram(
                    f"⏳ <b>Progress</b>: {i:,}/{total:,}\n"
                    f"✅ {success} · ↕️ {changed} · ⏭ {skipped}\n"
                    f"⏱ {elapsed//60}m {elapsed%60}s"
                )

            time.sleep(RATE_DELAY)

    elapsed = int(time.time() - start_time)
    summary = [
        f"✅ <b>Fetch Complete</b>", "",
        f"📋 Scanned:  <b>{total:,}</b>",
        f"✅ Found:    <b>{success:,}</b>",
        f"↕️ Changed:  <b>{changed}</b>",
        f"⏭ Skipped:  <b>{skipped:,}</b>",
        f"⏱ Duration: <b>{elapsed//60}m {elapsed%60}s</b>",
    ]
    if changes:
        summary += ["", "<b>Changes this run:</b>"]
        for line in changes[:25]:
            summary.append(f"  {line}")
        if len(changes) > 25:
            summary.append(f"  ... and {len(changes)-25} more")

    send_telegram("\n".join(summary))
    print(f"[fetch] Done. {success}/{total} found, {changed} changed, {skipped} skipped.")

if __name__ == "__main__":
    main()
