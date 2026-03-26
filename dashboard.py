import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import create_engine

st.set_page_config(page_title="DACA 821-D Tracker", page_icon="🛂", layout="wide")

DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

APPROVAL_KW  = ["approved","card was mailed","card is being produced",
                "card was delivered","employment authorization document approved","renewal approved"]
NEGATIVE_KW  = ["denied","rejected","terminated","administratively closed"]
NEEDS_ATT_KW = ["request for evidence","rfe","notice of intent to deny","noid"]

def classify(s):
    sl = str(s).lower()
    if any(k in sl for k in APPROVAL_KW):   return "Approved"
    if any(k in sl for k in NEGATIVE_KW):   return "Denied / Closed"
    if any(k in sl for k in NEEDS_ATT_KW):  return "Needs Attention"
    return "Pending"

STATUS_COLORS = {
    "Approved":        "#22c55e",
    "Pending":         "#3b82f6",
    "Needs Attention": "#f59e0b",
    "Denied / Closed": "#ef4444",
}

@st.cache_data(ttl=300)
def load():
    with engine.connect() as conn:
        cases = pd.read_sql("SELECT * FROM cases",      conn)
        logs  = pd.read_sql("SELECT * FROM status_log", conn)
        preds = pd.read_sql("SELECT * FROM predictions", conn)
    return cases, logs, preds

st.title("🛂 DACA Form I-821D Tracker")
st.caption(f"Supabase PostgreSQL · IOE · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

df, logs, pred = load()

if df.empty:
    st.warning("No cases yet. Run fetch_cases.py first.")
    st.stop()

df["outcome"] = df["last_status"].fillna("").apply(classify)

# ── KPIs ───────────────────────────────────────────────────────────────────
total     = len(df)
approved  = (df["outcome"] == "Approved").sum()
pending   = (df["outcome"] == "Pending").sum()
needs_att = (df["outcome"] == "Needs Attention").sum()
denied    = (df["outcome"] == "Denied / Closed").sum()

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total",           f"{total:,}")
k2.metric("Pending",         f"{pending:,}")
k3.metric("Approved",        f"{approved:,}", delta=f"{approved/total*100:.1f}%" if total else None)
k4.metric("Needs Attention", f"{needs_att:,}", delta_color="inverse")
k5.metric("Denied / Closed", f"{denied:,}",   delta_color="inverse")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Outcomes")
    oc = df["outcome"].value_counts().reset_index()
    oc.columns = ["outcome","count"]
    fig = px.pie(oc, values="count", names="outcome", hole=0.45,
                 color="outcome", color_discrete_map=STATUS_COLORS)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Pipeline stage")
    STAGE_MAP = {
        "approved":               "Approved / Mailed",
        "card was mailed":        "Approved / Mailed",
        "card is being produced": "Approved / Mailed",
        "denied":                 "Denied / Closed",
        "rejected":               "Denied / Closed",
        "terminated":             "Denied / Closed",
        "response to request":    "RFE response received",
        "request for evidence":   "RFE issued",
        "intent to deny":         "RFE issued",
        "interview":              "Interview scheduled",
        "actively reviewed":      "Under review",
        "biometric":              "Biometrics",
        "fingerprint":            "Biometrics",
        "received":               "Case received",
    }
    def map_stage(s):
        sl = str(s).lower()
        for kw, stage in STAGE_MAP.items():
            if kw in sl: return stage
        return "Other"

    df["stage"] = df["last_status"].apply(map_stage)
    sc = df["stage"].value_counts().reset_index()
    sc.columns = ["stage","count"]
    fig2 = px.bar(sc, x="count", y="stage", orientation="h",
                  color="count", color_continuous_scale="Blues")
    fig2.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300,
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Predictions ────────────────────────────────────────────────────────────
if not pred.empty:
    st.subheader("⏳ Estimated days remaining")
    mae = pred["model_mae_days"].median() if "model_mae_days" in pred.columns else None
    ph, ps = st.columns([2,1])
    with ph:
        fig3 = px.histogram(pred, x="est_days_remaining", nbins=40,
                            labels={"est_days_remaining":"Est. days remaining"},
                            color_discrete_sequence=["#3b82f6"])
        med = int(pred["est_days_remaining"].median())
        fig3.add_vline(x=med, line_dash="dash", line_color="orange",
                       annotation_text=f"Median: {med}d",
                       annotation_position="top right")
        fig3.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=280)
        st.plotly_chart(fig3, use_container_width=True)
    with ps:
        st.metric("Median days remaining",  med)
        st.metric("Decided in ≤30 days",   f"{(pred['est_days_remaining']<=30).sum():,}")
        st.metric("Decided in ≤60 days",   f"{(pred['est_days_remaining']<=60).sum():,}")
        if mae: st.metric("Model accuracy", f"±{mae:.0f} days")
    st.divider()

# ── Activity timeline ──────────────────────────────────────────────────────
if not logs.empty:
    st.subheader("📈 Daily fetch activity")
    logs["fetched_at"] = pd.to_datetime(logs["fetched_at"], utc=True, errors="coerce")
    daily = logs.groupby(logs["fetched_at"].dt.date).size().reset_index(name="records")
    daily.columns = ["date","records"]
    fig4 = px.area(daily, x="date", y="records",
                   color_discrete_sequence=["#3b82f6"])
    fig4.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=200)
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

# ── Needs attention ────────────────────────────────────────────────────────
att = df[df["outcome"]=="Needs Attention"]
if not att.empty:
    st.subheader(f"⚠️ Needs attention ({len(att):,})")
    st.dataframe(att[["receipt_number","last_status","action_date","updated_at"]],
                 use_container_width=True, height=220)
    st.divider()

# ── Case table ─────────────────────────────────────────────────────────────
st.subheader("🔎 Case lookup")
col_s, col_o = st.columns([3,1])
with col_s:
    search = st.text_input("Search receipt or status", placeholder="IOE249...")
with col_o:
    of = st.multiselect("Outcome", ["Approved","Pending","Needs Attention","Denied / Closed"])

display = df.copy()
if not pred.empty:
    display = display.merge(
        pred[["receipt_number","est_days_remaining","confidence"]],
        on="receipt_number", how="left")
if search:
    mask = (display["receipt_number"].str.contains(search, case=False, na=False) |
            display["last_status"].str.contains(search, case=False, na=False))
    display = display[mask]
if of:
    display = display[display["outcome"].isin(of)]

cols  = ["receipt_number","form_type","last_status","stage","outcome","action_date","updated_at"]
extra = [c for c in ["est_days_remaining","confidence"] if c in display.columns]
st.dataframe(display[cols+extra].rename(columns={
    "receipt_number":"Receipt","form_type":"Form","last_status":"Status",
    "stage":"Stage","outcome":"Outcome","action_date":"Action date",
    "updated_at":"Last fetched","est_days_remaining":"Est. days left",
    "confidence":"Confidence",
}), use_container_width=True, height=420)

csv = display[cols+extra].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "daca_cases.csv", "text/csv")
