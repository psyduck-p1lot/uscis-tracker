import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="USCIS 821-D Tracker",
    page_icon="🛂",
    layout="wide",
)

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_cases():
    r = sb.table("cases").select("*").execute()
    return pd.DataFrame(r.data)

@st.cache_data(ttl=300)
def load_status_log():
    r = sb.table("status_log").select("*").execute()
    return pd.DataFrame(r.data)

@st.cache_data(ttl=300)
def load_predictions():
    r = sb.table("predictions").select("*").execute()
    return pd.DataFrame(r.data)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛂 USCIS Form 821-D Case Tracker")
st.caption(f"Data refreshes every 5 minutes · Last loaded: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

df_cases = load_cases()
df_log   = load_status_log()
df_pred  = load_predictions()

if df_cases.empty:
    st.warning("No cases in database yet. Run fetch_cases.py first.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
APPROVAL_KW = ["approved", "card was mailed", "card is being produced"]
def is_approved(s):
    return any(kw in str(s).lower() for kw in APPROVAL_KW)

total    = len(df_cases)
approved = df_cases["last_status"].apply(is_approved).sum()
pending  = total - approved
changed  = 0
if not df_log.empty:
    counts = df_log.groupby("receipt_number")["status"].nunique()
    changed = int((counts > 1).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cases",   total)
c2.metric("Pending",       pending)
c3.metric("Approved",      approved,  delta=f"{round(approved/total*100, 1)}%" if total else None)
c4.metric("Ever Changed",  changed)

st.divider()

# ── Row 1: Status breakdown + Service center ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Status breakdown")
    status_counts = df_cases["last_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.bar(
        status_counts, x="count", y="status", orientation="h",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Cases", "status": ""},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=0, b=0), height=320)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("By service center")
    sc_counts = df_cases["service_center"].value_counts().reset_index()
    sc_counts.columns = ["center", "count"]
    fig2 = px.pie(sc_counts, values="count", names="center", hole=0.4)
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=320)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Prediction histogram ───────────────────────────────────────────────
if not df_pred.empty:
    st.subheader("⏳ Estimated days remaining (pending cases)")
    mae = df_pred["model_mae_days"].iloc[0] if "model_mae_days" in df_pred.columns else None
    if mae:
        st.caption(f"Model accuracy: ±{mae:.0f} days MAE · Based on resolved cases in this dataset")

    fig3 = px.histogram(
        df_pred, x="est_days_remaining", nbins=30,
        labels={"est_days_remaining": "Estimated days remaining"},
        color_discrete_sequence=["#4f8ef7"],
    )
    fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
    st.plotly_chart(fig3, use_container_width=True)

    soon = df_pred[df_pred["est_days_remaining"] <= 30].shape[0]
    if soon:
        st.info(f"🔔 **{soon} cases** predicted to receive a decision within 30 days.")

    st.divider()

# ── Row 3: Status timeline ────────────────────────────────────────────────────
if not df_log.empty:
    st.subheader("📈 Status activity over time")
    df_log["fetched_at"] = pd.to_datetime(df_log["fetched_at"], utc=True, errors="coerce")
    daily = (
        df_log.groupby(df_log["fetched_at"].dt.date)
        .size()
        .reset_index(name="updates")
    )
    daily.columns = ["date", "updates"]
    fig4 = px.line(daily, x="date", y="updates",
                   labels={"date": "", "updates": "Status records fetched"},
                   color_discrete_sequence=["#4f8ef7"])
    fig4.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220)
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

# ── Row 4: Case table ─────────────────────────────────────────────────────────
st.subheader("🔎 Case lookup")

search = st.text_input("Search by receipt number or status", placeholder="e.g. IOR2490...")
sc_filter = st.multiselect(
    "Filter by service center",
    options=sorted(df_cases["service_center"].dropna().unique()),
)

display = df_cases.copy()
if not df_pred.empty:
    display = display.merge(
        df_pred[["receipt_number", "est_days_remaining", "confidence"]],
        on="receipt_number", how="left"
    )

if search:
    mask = (
        display["receipt_number"].str.contains(search, case=False, na=False) |
        display["last_status"].str.contains(search, case=False, na=False)
    )
    display = display[mask]

if sc_filter:
    display = display[display["service_center"].isin(sc_filter)]

show_cols = ["receipt_number", "form_type", "service_center",
             "last_status", "action_date", "updated_at"]
if "est_days_remaining" in display.columns:
    show_cols += ["est_days_remaining", "confidence"]

st.dataframe(
    display[show_cols].rename(columns={
        "receipt_number":    "Receipt",
        "form_type":         "Form",
        "service_center":    "Center",
        "last_status":       "Status",
        "action_date":       "Action date",
        "updated_at":        "Last fetched",
        "est_days_remaining":"Est. days left",
        "confidence":        "Confidence",
    }),
    use_container_width=True,
    height=400,
)

# CSV export
csv = display[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "uscis_cases.csv", "text/csv")
