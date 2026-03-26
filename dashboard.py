import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client
from datetime import datetime, timezone

st.set_page_config(page_title="DACA 821-D Tracker", page_icon="🛂", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

APPROVAL_KW = [
    "approved", "card was mailed", "card is being produced",
    "card was delivered", "employment authorization document approved", "renewal approved",
]
NEGATIVE_KW  = ["denied", "rejected", "terminated", "administratively closed"]
NEEDS_ATT_KW = ["request for evidence", "rfe", "notice of intent to deny", "noid"]

def classify(s: str) -> str:
    sl = s.lower()
    if any(k in sl for k in APPROVAL_KW):   return "Approved"
    if any(k in sl for k in NEGATIVE_KW):   return "Denied / Closed"
    if any(k in sl for k in NEEDS_ATT_KW):  return "Needs Attention"
    return "Pending"

STATUS_COLORS = {
    "Approved":         "#22c55e",
    "Pending":          "#3b82f6",
    "Needs Attention":  "#f59e0b",
    "Denied / Closed":  "#ef4444",
}

@st.cache_data(ttl=300)
def load_cases():
    return pd.DataFrame(sb.table("cases").select("*").execute().data)

@st.cache_data(ttl=300)
def load_logs():
    return pd.DataFrame(sb.table("status_log").select("*").execute().data)

@st.cache_data(ttl=300)
def load_predictions():
    return pd.DataFrame(sb.table("predictions").select("*").execute().data)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🛂 DACA Form I-821D Case Tracker")
st.caption(
    f"IOE (ELIS) service center · Data refreshes every 5 min · "
    f"Last loaded: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)

df   = load_cases()
logs = load_logs()
pred = load_predictions()

if df.empty:
    st.warning("No cases found yet. Run fetch_cases.py first.")
    st.stop()

df["outcome"] = df["last_status"].fillna("").apply(classify)

# ── KPI row ────────────────────────────────────────────────────────────────
total      = len(df)
approved   = (df["outcome"] == "Approved").sum()
pending    = (df["outcome"] == "Pending").sum()
needs_att  = (df["outcome"] == "Needs Attention").sum()
denied     = (df["outcome"] == "Denied / Closed").sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total cases",      f"{total:,}")
k2.metric("Pending",          f"{pending:,}")
k3.metric("Approved",         f"{approved:,}",
          delta=f"{approved/total*100:.1f}%" if total else None)
k4.metric("Needs attention",  f"{needs_att:,}",
          delta=f"{needs_att} RFE/NOID" if needs_att else None,
          delta_color="inverse")
k5.metric("Denied / Closed",  f"{denied:,}",
          delta_color="inverse")

st.divider()

# ── Row 1: Outcome pie + pipeline stage bar ───────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Case outcomes")
    outcome_df = df["outcome"].value_counts().reset_index()
    outcome_df.columns = ["outcome", "count"]
    fig = px.pie(
        outcome_df, values="count", names="outcome", hole=0.45,
        color="outcome", color_discrete_map=STATUS_COLORS,
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("DACA pipeline stage")
    stage_order = [
        "Case received", "Biometrics scheduled", "Biometrics taken",
        "Under review", "RFE issued", "RFE response received",
        "Interview scheduled", "Card being produced", "Approved / Mailed",
        "Denied / Closed", "Other",
    ]
    def map_stage(s: str) -> str:
        sl = s.lower()
        if "card is being produced" in sl or "card was mailed" in sl or "approved" in sl: return "Approved / Mailed"
        if "denied" in sl or "rejected" in sl or "terminated" in sl:                      return "Denied / Closed"
        if "response to request" in sl:                                                    return "RFE response received"
        if "request for evidence" in sl or "rfe" in sl or "intent to deny" in sl:         return "RFE issued"
        if "interview" in sl:                                                              return "Interview scheduled"
        if "actively reviewed" in sl or "updated" in sl:                                  return "Under review"
        if "biometric" in sl or "fingerprint" in sl:
            return "Biometrics taken" if "taken" in sl or "were taken" in sl else "Biometrics scheduled"
        if "received" in sl:                                                               return "Case received"
        return "Other"

    df["stage"] = df["last_status"].fillna("").apply(map_stage)
    stage_df = df["stage"].value_counts().reindex(stage_order, fill_value=0).reset_index()
    stage_df.columns = ["stage", "count"]
    stage_df = stage_df[stage_df["count"] > 0]
    fig2 = px.bar(
        stage_df, x="count", y="stage", orientation="h",
        color="count", color_continuous_scale="Blues",
    )
    fig2.update_layout(
        margin=dict(l=0,r=0,t=0,b=0), height=300,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="array", categoryarray=stage_order[::-1]),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Prediction histogram + stats ───────────────────────────────────
if not pred.empty:
    st.subheader("⏳ Estimated days remaining to decision")

    mae = pred["model_mae_days"].median() if "model_mae_days" in pred.columns else None
    rfe_cases   = pred["had_rfe"].sum()   if "had_rfe"   in pred.columns else 0
    renewal_cnt = pred["is_renewal"].sum() if "is_renewal" in pred.columns else 0

    col_h, col_s = st.columns([2, 1])
    with col_h:
        fig3 = px.histogram(
            pred, x="est_days_remaining", nbins=40,
            labels={"est_days_remaining": "Estimated days remaining"},
            color_discrete_sequence=["#3b82f6"],
        )
        if mae:
            fig3.add_vline(
                x=pred["est_days_remaining"].median(),
                line_dash="dash", line_color="orange",
                annotation_text=f"Median: {int(pred['est_days_remaining'].median())}d",
                annotation_position="top right",
            )
        fig3.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with col_s:
        st.metric("Median days remaining", f"{int(pred['est_days_remaining'].median())}")
        st.metric("Decided in ≤30 days",   f"{(pred['est_days_remaining'] <= 30).sum():,}")
        st.metric("Decided in ≤60 days",   f"{(pred['est_days_remaining'] <= 60).sum():,}")
        if mae:
            st.metric("Model accuracy",    f"±{mae:.0f} days MAE")
        if rfe_cases:
            st.metric("Pending w/ prior RFE", f"{int(rfe_cases):,}")
        if renewal_cnt:
            st.metric("Renewals pending",  f"{int(renewal_cnt):,}")

    soon = (pred["est_days_remaining"] <= 30).sum()
    if soon:
        st.info(f"🔔 **{soon:,} cases** predicted to receive a decision within 30 days.")

    st.divider()

# ── Row 3: Status change timeline ─────────────────────────────────────────
if not logs.empty:
    st.subheader("📈 Daily status activity")
    logs["fetched_at"] = pd.to_datetime(logs["fetched_at"], utc=True, errors="coerce")
    daily = (
        logs.groupby(logs["fetched_at"].dt.date).size()
        .reset_index(name="records")
        .rename(columns={"fetched_at": "date"})
    )
    fig4 = px.area(
        daily, x="date", y="records",
        labels={"date": "", "records": "Status records fetched"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig4.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=200)
    st.plotly_chart(fig4, use_container_width=True)

    # RFE trend
    rfe_logs = logs[logs["status"].str.lower().str.contains("request for evidence|rfe", na=False)]
    if not rfe_logs.empty:
        rfe_daily = (
            rfe_logs.groupby(rfe_logs["fetched_at"].dt.date).size()
            .reset_index(name="rfe_count")
            .rename(columns={"fetched_at": "date"})
        )
        with st.expander("📊 RFE issuance trend"):
            fig5 = px.bar(
                rfe_daily, x="date", y="rfe_count",
                labels={"rfe_count": "RFEs detected", "date": ""},
                color_discrete_sequence=["#f59e0b"],
            )
            fig5.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=200)
            st.plotly_chart(fig5, use_container_width=True)

    st.divider()

# ── Row 4: Needs attention callout ────────────────────────────────────────
att_df = df[df["outcome"] == "Needs Attention"]
if not att_df.empty:
    st.subheader(f"⚠️ Cases needing attention ({len(att_df):,})")
    st.caption("RFE issued, NOID, or other action required")
    show = att_df[["receipt_number","last_status","action_date","updated_at"]].rename(columns={
        "receipt_number": "Receipt",
        "last_status":    "Status",
        "action_date":    "Action date",
        "updated_at":     "Last fetched",
    })
    st.dataframe(show, use_container_width=True, height=220)
    st.divider()

# ── Row 5: Full case table ─────────────────────────────────────────────────
st.subheader("🔎 Case lookup")

col_s, col_o = st.columns([3, 1])
with col_s:
    search = st.text_input("Search receipt number or status", placeholder="e.g. IOE2490...")
with col_o:
    outcome_filter = st.multiselect(
        "Filter by outcome",
        options=["Approved", "Pending", "Needs Attention", "Denied / Closed"],
    )

display = df.copy()
if not pred.empty:
    display = display.merge(
        pred[["receipt_number","est_days_remaining","confidence","is_renewal","had_rfe"]],
        on="receipt_number", how="left",
    )
if search:
    mask = (
        display["receipt_number"].str.contains(search, case=False, na=False) |
        display["last_status"].str.contains(search, case=False, na=False)
    )
    display = display[mask]
if outcome_filter:
    display = display[display["outcome"].isin(outcome_filter)]

base_cols = ["receipt_number","form_type","last_status","stage","outcome","action_date","updated_at"]
extra_cols = [c for c in ["est_days_remaining","confidence","is_renewal","had_rfe"] if c in display.columns]
show_cols  = base_cols + extra_cols

rename_map = {
    "receipt_number":     "Receipt",
    "form_type":          "Form",
    "last_status":        "Status",
    "stage":              "Pipeline stage",
    "outcome":            "Outcome",
    "action_date":        "Action date",
    "updated_at":         "Last fetched",
    "est_days_remaining": "Est. days left",
    "confidence":         "Confidence",
    "is_renewal":         "Renewal?",
    "had_rfe":            "Had RFE?",
}

st.dataframe(
    display[show_cols].rename(columns=rename_map),
    use_container_width=True,
    height=420,
)

csv = display[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "daca_cases.csv", "text/csv")
