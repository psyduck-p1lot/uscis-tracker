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

APPROVAL_KEYWORDS = ["approved", "card was mailed", "card is being produced"]
MIN_TRAINING_ROWS = 30   # skip training if not enough resolved cases yet

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram] {e}")

def is_approved(status: str) -> bool:
    s = status.lower()
    return any(kw in s for kw in APPROVAL_KEYWORDS)

def load_status_log() -> pd.DataFrame:
    res = sb.table("status_log").select("*").execute()
    return pd.DataFrame(res.data)

def load_cases() -> pd.DataFrame:
    res = sb.table("cases").select("*").execute()
    return pd.DataFrame(res.data)

def build_features(df_log: pd.DataFrame, df_cases: pd.DataFrame) -> pd.DataFrame:
    df_log["fetched_at"] = pd.to_datetime(df_log["fetched_at"], utc=True)
    df_log["action_date"] = pd.to_datetime(df_log["action_date"], utc=True, errors="coerce")

    # First seen date per receipt (proxy for filing date if no priority_date)
    first_seen = (
        df_log.groupby("receipt_number")["fetched_at"]
        .min()
        .rename("first_seen")
    )

    # Approval date (if resolved)
    approval = df_log[df_log["status"].str.lower().apply(is_approved)].copy()
    approval_date = (
        approval.groupby("receipt_number")["fetched_at"]
        .min()
        .rename("approved_at")
    )

    feat = df_cases.merge(first_seen, on="receipt_number", how="left")
    feat = feat.merge(approval_date, on="receipt_number", how="left")

    feat["first_seen"]  = pd.to_datetime(feat["first_seen"],  utc=True, errors="coerce")
    feat["approved_at"] = pd.to_datetime(feat["approved_at"], utc=True, errors="coerce")

    now = datetime.now(timezone.utc)

    # Days from first seen to approval (label for training)
    feat["days_to_approval"] = (feat["approved_at"] - feat["first_seen"]).dt.days

    # Days elapsed since first seen
    feat["days_elapsed"] = (now - feat["first_seen"]).dt.days.fillna(0).astype(int)

    # Filing month (seasonality signal)
    feat["filing_month"] = feat["first_seen"].dt.month.fillna(0).astype(int)

    # Status progression score: higher = further along pipeline
    progression_map = {
        "received":    1,
        "biometrics":  2,
        "fingerprint": 2,
        "under review":3,
        "interview":   4,
        "approved":    5,
        "card":        5,
        "denied":      5,
        "rejected":    5,
    }
    def score_status(s):
        if not isinstance(s, str):
            return 0
        sl = s.lower()
        for kw, score in progression_map.items():
            if kw in sl:
                return score
        return 1

    feat["status_score"] = feat["last_status"].apply(score_status)

    # Encode service center
    le = LabelEncoder()
    feat["sc_encoded"] = le.fit_transform(feat["service_center"].fillna("UNK"))

    return feat

def train_and_predict(feat: pd.DataFrame):
    resolved   = feat[feat["days_to_approval"].notna()].copy()
    unresolved = feat[feat["days_to_approval"].isna()].copy()

    feature_cols = ["days_elapsed", "filing_month", "status_score", "sc_encoded"]

    if len(resolved) < MIN_TRAINING_ROWS:
        send_telegram(
            f"⚠️ <b>Prediction skipped</b>\n"
            f"Only <b>{len(resolved)}</b> resolved cases — need {MIN_TRAINING_ROWS} to train.\n"
            f"Predictions will improve as more cases resolve."
        )
        print(f"[predict] Not enough resolved cases ({len(resolved)}). Skipping.")
        return

    X = resolved[feature_cols]
    y = resolved["days_to_approval"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)

    mae = mean_absolute_error(y_te, model.predict(X_te))
    print(f"[predict] Model MAE: {mae:.1f} days (on {len(resolved)} resolved cases)")

    if len(unresolved) == 0:
        send_telegram(f"🤖 <b>Model trained</b> — MAE: <b>{mae:.0f} days</b>\nNo pending cases to predict.")
        return

    X_pred = unresolved[feature_cols]
    raw_preds = model.predict(X_pred)

    rows = []
    now  = datetime.now(timezone.utc).isoformat()
    for receipt, days_elapsed, pred_total in zip(
        unresolved["receipt_number"], unresolved["days_elapsed"], raw_preds
    ):
        est_remaining = max(0, int(round(pred_total - days_elapsed)))
        confidence    = max(0.0, min(1.0, 1.0 - (mae / max(pred_total, 1))))
        rows.append({
            "receipt_number":   receipt,
            "est_days_remaining": est_remaining,
            "est_total_days":   int(round(pred_total)),
            "model_mae_days":   round(mae, 1),
            "confidence":       round(confidence, 2),
            "predicted_at":     now,
        })

    # Upsert predictions
    sb.table("predictions").upsert(rows, on_conflict="receipt_number").execute()

    # Stats for Telegram
    est_arr   = np.array([r["est_days_remaining"] for r in rows])
    med_days  = int(np.median(est_arr))
    min_days  = int(est_arr.min())
    max_days  = int(est_arr.max())
    soon_30   = int((est_arr <= 30).sum())

    send_telegram(
        f"🤖 <b>Prediction Model Updated</b>\n"
        f"\n"
        f"📊 Trained on <b>{len(resolved)}</b> resolved cases\n"
        f"🎯 Model accuracy: ±<b>{mae:.0f} days</b> MAE\n"
        f"\n"
        f"⏳ <b>Pending cases: {len(unresolved)}</b>\n"
        f"  Median est. remaining: <b>{med_days} days</b>\n"
        f"  Range: {min_days}–{max_days} days\n"
        f"  Likely approved in ≤30 days: <b>{soon_30}</b>"
    )
    print(f"[predict] Wrote {len(rows)} predictions. Median remaining: {med_days} days.")

def main():
    print("[predict] Loading data...")
    df_log   = load_status_log()
    df_cases = load_cases()

    if df_log.empty or df_cases.empty:
        send_telegram("⚠️ <b>Prediction skipped</b> — no data in database yet.")
        return

    feat = build_features(df_log, df_cases)
    train_and_predict(feat)

if __name__ == "__main__":
    main()
