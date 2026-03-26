import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

load_dotenv()

TG_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
DATABASE_URL = os.environ["DATABASE_URL"]

engine   = create_engine(DATABASE_URL, pool_pre_ping=True)
MIN_ROWS = 30

APPROVAL_KW = [
    "approved", "card was mailed", "card is being produced",
    "card was delivered", "employment authorization document approved", "renewal approved",
]
NEGATIVE_KW = ["denied", "rejected", "terminated", "administratively closed"]

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg,
                                     "parse_mode": "HTML"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram] {e}")

def is_approved(s): return any(k in str(s).lower() for k in APPROVAL_KW)
def is_negative(s): return any(k in str(s).lower() for k in NEGATIVE_KW)

def load_data():
    with engine.connect() as conn:
        cases = pd.read_sql("SELECT * FROM cases",      conn)
        logs  = pd.read_sql("SELECT * FROM status_log", conn)
    return cases, logs

def build_features(cases, logs):
    logs["fetched_at"] = pd.to_datetime(logs["fetched_at"], utc=True, errors="coerce")

    first_seen    = logs.groupby("receipt_number")["fetched_at"].min().rename("first_seen")
    approval_date = (
        logs[logs["status"].apply(is_approved)]
        .groupby("receipt_number")["fetched_at"].min().rename("approved_at")
    )

    feat = cases.merge(first_seen, on="receipt_number", how="left")
    feat = feat.merge(approval_date, on="receipt_number", how="left")

    feat["first_seen"]  = pd.to_datetime(feat["first_seen"],  utc=True, errors="coerce")
    feat["approved_at"] = pd.to_datetime(feat["approved_at"], utc=True, errors="coerce")

    now = datetime.now(timezone.utc)
    feat["days_to_approval"] = (feat["approved_at"] - feat["first_seen"]).dt.days
    feat["days_elapsed"]     = (now - feat["first_seen"]).dt.days.fillna(0).astype(int)
    feat["filing_month"]     = feat["first_seen"].dt.month.fillna(0).astype(int)
    feat["filing_year"]      = feat["first_seen"].dt.year.fillna(0).astype(int)
    feat["pipeline_score"]   = pd.to_numeric(feat.get("pipeline_score", 0), errors="coerce").fillna(0).astype(int)

    def had_rfe(r):
        return int(logs[logs["receipt_number"]==r]["status"].str.lower().str.contains("request for evidence").any())
    def had_bio(r):
        return int(logs[logs["receipt_number"]==r]["status"].str.lower().str.contains("biometric|fingerprint").any())
    def is_renewal(r):
        return int(logs[logs["receipt_number"]==r]["status"].str.lower().str.contains("renewal").any())

    feat["had_rfe"]        = feat["receipt_number"].apply(had_rfe)
    feat["had_biometrics"] = feat["receipt_number"].apply(had_bio)
    feat["is_renewal"]     = feat["receipt_number"].apply(is_renewal)

    transitions = logs.groupby("receipt_number")["status"].nunique().rename("status_transitions")
    feat = feat.merge(transitions, on="receipt_number", how="left")
    feat["status_transitions"] = feat["status_transitions"].fillna(1).astype(int)

    return feat

def train_and_predict(feat):
    resolved = feat[feat["days_to_approval"].notna() & ~feat["last_status"].apply(is_negative)].copy()
    pending  = feat[feat["days_to_approval"].isna()  & ~feat["last_status"].apply(is_negative)].copy()

    FEATURES = ["days_elapsed", "filing_month", "filing_year", "pipeline_score",
                "had_rfe", "had_biometrics", "is_renewal", "status_transitions"]

    if len(resolved) < MIN_ROWS:
        send_telegram(
            f"⚠️ <b>Prediction skipped</b>\n"
            f"Only <b>{len(resolved)}</b> resolved cases (need {MIN_ROWS}).\n"
            f"Predictions unlock automatically as more cases resolve."
        )
        print(f"[predict] Not enough resolved cases ({len(resolved)}). Skipping.")
        return

    X, y = resolved[FEATURES], resolved["days_to_approval"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, random_state=42
    )
    model.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, model.predict(X_te))
    print(f"[predict] MAE={mae:.1f} days on {len(resolved)} resolved cases")

    if pending.empty:
        send_telegram(f"🤖 <b>Model trained</b> — ±{mae:.0f} days MAE\nNo pending cases to score.")
        return

    preds = model.predict(pending[FEATURES])
    now   = datetime.now(timezone.utc).isoformat()
    rows  = []

    for receipt, days_elapsed, rfe, renewal, pred_total in zip(
        pending["receipt_number"], pending["days_elapsed"],
        pending["had_rfe"], pending["is_renewal"], preds
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

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text("""
                INSERT INTO predictions
                    (receipt_number, est_days_remaining, est_total_days,
                     model_mae_days, confidence, had_rfe, is_renewal, predicted_at)
                VALUES
                    (:receipt_number, :est_days_remaining, :est_total_days,
                     :model_mae_days, :confidence, :had_rfe, :is_renewal, :predicted_at)
                ON CONFLICT (receipt_number) DO UPDATE SET
                    est_days_remaining = EXCLUDED.est_days_remaining,
                    est_total_days     = EXCLUDED.est_total_days,
                    model_mae_days     = EXCLUDED.model_mae_days,
                    confidence         = EXCLUDED.confidence,
                    had_rfe            = EXCLUDED.had_rfe,
                    is_renewal         = EXCLUDED.is_renewal,
                    predicted_at       = EXCLUDED.predicted_at
            """), row)

    est_arr = np.array([r["est_days_remaining"] for r in rows])
    median  = int(np.median(est_arr))
    p25     = int(np.percentile(est_arr, 25))
    p75     = int(np.percentile(est_arr, 75))

    send_telegram(
        f"🤖 <b>DACA Prediction Updated</b>\n\n"
        f"📊 Trained on <b>{len(resolved)}</b> resolved cases\n"
        f"🎯 Accuracy: ±<b>{mae:.0f} days</b>\n\n"
        f"⏳ <b>Pending: {len(pending):,}</b>\n"
        f"  Median remaining: <b>{median} days</b>\n"
        f"  Middle 50%: {p25}–{p75} days\n"
        f"  ≤30 days: <b>{(est_arr<=30).sum()}</b> · "
        f"≤60 days: <b>{(est_arr<=60).sum()}</b>"
    )
    print(f"[predict] Wrote {len(rows)} predictions. Median: {median} days.")

def main():
    print("[predict] Loading data...")
    cases, logs = load_data()
    if cases.empty or logs.empty:
        send_telegram("⚠️ <b>Prediction skipped</b> — no data yet. Run fetch first.")
        return
    feat = build_features(cases, logs)
    train_and_predict(feat)

if __name__ == "__main__":
    main()
