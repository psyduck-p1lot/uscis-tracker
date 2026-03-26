import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TG_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

MIN_TRAINING_ROWS = 30

# DACA pipeline stages in order — used to compute progression velocity
PIPELINE_ORDER = [
    "case was received",
    "fees were received",
    "biometrics appointment was scheduled",
    "biometrics were taken",
    "fingerprints were taken",
    "case is being actively reviewed",
    "request for evidence",
    "response to request for evidence received",
    "interview was scheduled",
    "card is being produced",
    "card was mailed",
    "case was approved",
    "employment authorization document approved",
    "renewal approved",
]

APPROVAL_KW = [
    "approved", "card was mailed", "card is being produced",
    "card was delivered", "employment authorization document approved",
    "renewal approved",
]

NEGATIVE_KW = ["denied", "rejected", "terminated", "administratively closed"]

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

def is_approved(s: str) -> bool:
    return any(kw in s.lower() for kw in APPROVAL_KW)

def is_negative(s: str) -> bool:
    return any(kw in s.lower() for kw in NEGATIVE_KW)

def had_rfe(logs: pd.DataFrame, receipt: str) -> int:
    rows = logs[logs["receipt_number"] == receipt]
    return int(rows["status"].str.lower().str.contains("request for evidence").any())

def had_biometrics(logs: pd.DataFrame, receipt: str) -> int:
    rows = logs[logs["receipt_number"] == receipt]
    return int(rows["status"].str.lower().str.contains("biometric|fingerprint").any())

def is_renewal(logs: pd.DataFrame, receipt: str) -> int:
    """Renewals tend to have 'renewal' in a status at some point."""
    rows = logs[logs["receipt_number"] == receipt]
    return int(rows["status"].str.lower().str.contains("renewal").any())

def days_at_biometrics(logs: pd.DataFrame, receipt: str) -> int:
    """Days between biometrics taken and the next status update — proxy for processing delay."""
    rows = logs[logs["receipt_number"] == receipt].sort_values("fetched_at")
    bio_rows = rows[rows["status"].str.lower().str.contains("biometric|fingerprint")]
    if bio_rows.empty:
        return 0
    bio_date = pd.to_datetime(bio_rows["fetched_at"].iloc[0], utc=True)
    later = rows[rows["fetched_at"] > bio_rows["fetched_at"].iloc[0]]
    if later.empty:
        return 0
    next_date = pd.to_datetime(later["fetched_at"].iloc[0], utc=True)
    return max(0, (next_date - bio_date).days)

def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.DataFrame(sb.table("cases").select("*").execute().data)
    logs  = pd.DataFrame(sb.table("status_log").select("*").execute().data)
    return cases, logs

def build_features(cases: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    logs["fetched_at"]  = pd.to_datetime(logs["fetched_at"],  utc=True, errors="coerce")
    logs["action_date"] = pd.to_datetime(logs["action_date"], utc=True, errors="coerce")

    first_seen = (
        logs.groupby("receipt_number")["fetched_at"].min().rename("first_seen")
    )
    approval_date = (
        logs[logs["status"].apply(is_approved)]
        .groupby("receipt_number")["fetched_at"]
        .min()
        .rename("approved_at")
    )

    feat = cases.merge(first_seen, on="receipt_number", how="left")
    feat = feat.merge(approval_date, on="receipt_number", how="left")

    feat["first_seen"]  = pd.to_datetime(feat["first_seen"],  utc=True, errors="coerce")
    feat["approved_at"] = pd.to_datetime(feat["approved_at"], utc=True, errors="coerce")

    now = datetime.now(timezone.utc)

    # Label: total days from first seen to approval
    feat["days_to_approval"] = (feat["approved_at"] - feat["first_seen"]).dt.days

    # Feature: days elapsed since first seen
    feat["days_elapsed"] = (now - feat["first_seen"]).dt.days.fillna(0).astype(int)

    # Feature: filing month (1–12) — USCIS processing times vary by intake volume
    feat["filing_month"] = feat["first_seen"].dt.month.fillna(0).astype(int)

    # Feature: filing year — policy changes affect timelines year to year
    feat["filing_year"]  = feat["first_seen"].dt.year.fillna(0).astype(int)

    # Feature: current pipeline score from fetch_cases taxonomy
    feat["pipeline_score"] = pd.to_numeric(feat.get("pipeline_score", 0), errors="coerce").fillna(0).astype(int)

    # DACA-specific features
    feat["had_rfe"]         = feat["receipt_number"].apply(lambda r: had_rfe(logs, r))
    feat["had_biometrics"]  = feat["receipt_number"].apply(lambda r: had_biometrics(logs, r))
    feat["is_renewal"]      = feat["receipt_number"].apply(lambda r: is_renewal(logs, r))
    feat["bio_gap_days"]    = feat["receipt_number"].apply(lambda r: days_at_biometrics(logs, r))

    # Feature: number of distinct statuses seen (more transitions = more data = better signal)
    transition_counts = (
        logs.groupby("receipt_number")["status"].nunique().rename("status_transitions")
    )
    feat = feat.merge(transition_counts, on="receipt_number", how="left")
    feat["status_transitions"] = feat["status_transitions"].fillna(1).astype(int)

    return feat

def train_and_predict(feat: pd.DataFrame, logs: pd.DataFrame):
    # Exclude negative outcomes from training — they skew approval timeline
    resolved = feat[
        feat["days_to_approval"].notna() &
        ~feat["last_status"].apply(is_negative)
    ].copy()

    pending = feat[
        feat["days_to_approval"].isna() &
        ~feat["last_status"].apply(is_negative)
    ].copy()

    feature_cols = [
        "days_elapsed",
        "filing_month",
        "filing_year",
        "pipeline_score",
        "had_rfe",
        "had_biometrics",
        "is_renewal",
        "bio_gap_days",
        "status_transitions",
    ]

    if len(resolved) < MIN_TRAINING_ROWS:
        send_telegram(
            f"⚠️ <b>Prediction skipped</b>\n"
            f"Only <b>{len(resolved)}</b> resolved DACA cases so far.\n"
            f"Need <b>{MIN_TRAINING_ROWS}</b> to train the model.\n"
            f"Keep running daily — predictions unlock automatically."
        )
        print(f"[predict] Not enough resolved cases ({len(resolved)}). Skipping.")
        return

    X = resolved[feature_cols]
    y = resolved["days_to_approval"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, random_state=42
    )
    model.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, model.predict(X_te))
    print(f"[predict] MAE: {mae:.1f} days on {len(resolved)} resolved cases")

    if pending.empty:
        send_telegram(
            f"🤖 <b>DACA Model Trained</b>\n"
            f"Resolved cases: <b>{len(resolved)}</b>\n"
            f"Accuracy: ±<b>{mae:.0f} days</b> MAE\n"
            f"No pending cases to score."
        )
        return

    X_pred     = pending[feature_cols]
    raw_preds  = model.predict(X_pred)

    rows = []
    now  = datetime.now(timezone.utc).isoformat()

    for receipt, days_elapsed, rfe, renewal, pred_total in zip(
        pending["receipt_number"],
        pending["days_elapsed"],
        pending["had_rfe"],
        pending["is_renewal"],
        raw_preds,
    ):
        est_remaining = max(0, int(round(pred_total - days_elapsed)))
        confidence    = round(max(0.0, min(1.0, 1.0 - (mae / max(pred_total, 1)))), 2)
        rows.append({
            "receipt_number":     receipt,
            "est_days_remaining": est_remaining,
            "est_total_days":     int(round(pred_total)),
            "model_mae_days":     round(mae, 1),
            "confidence":         confidence,
            "had_rfe":            bool(rfe),
            "is_renewal":         bool(renewal),
            "predicted_at":       now,
        })

    sb.table("predictions").upsert(rows, on_conflict="receipt_number").execute()

    est_arr  = np.array([r["est_days_remaining"] for r in rows])
    median   = int(np.median(est_arr))
    p25      = int(np.percentile(est_arr, 25))
    p75      = int(np.percentile(est_arr, 75))
    soon_30  = int((est_arr <= 30).sum())
    soon_60  = int((est_arr <= 60).sum())
    rfe_pend = sum(1 for r in rows if r["had_rfe"])
    renewals = sum(1 for r in rows if r["is_renewal"])

    send_telegram(
        f"🤖 <b>DACA Prediction Model Updated</b>\n"
        f"\n"
        f"📊 Trained on <b>{len(resolved)}</b> resolved cases\n"
        f"🎯 Accuracy: ±<b>{mae:.0f} days</b> MAE\n"
        f"\n"
        f"⏳ <b>Pending cases: {len(pending):,}</b>\n"
        f"  Median est. remaining: <b>{median} days</b>\n"
        f"  Middle 50%: {p25}–{p75} days\n"
        f"  Likely decided in ≤30 days: <b>{soon_30}</b>\n"
        f"  Likely decided in ≤60 days: <b>{soon_60}</b>\n"
        f"\n"
        f"📋 Case mix:\n"
        f"  Renewals: <b>{renewals}</b>\n"
        f"  Cases with prior RFE: <b>{rfe_pend}</b>"
    )
    print(f"[predict] Wrote {len(rows)} predictions. Median remaining: {median} days.")

def main():
    print("[predict] Loading data from Supabase...")
    cases, logs = load_all()

    if cases.empty or logs.empty:
        send_telegram("⚠️ <b>Prediction skipped</b> — no data yet. Run fetch first.")
        return

    feat = build_features(cases, logs)
    train_and_predict(feat, logs)

if __name__ == "__main__":
    main()
