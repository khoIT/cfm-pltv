"""
Page 5 — Feedback & Learning
Time dynamics, robustness/stability checks, and A/B test planning.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import render_sidebar, get_data, format_currency, convert_vnd, get_currency_info, REPORTS_DIR

render_sidebar()

st.title("🔄 Layer 5 — Feedback & Learning")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page in the sidebar.")
    st.stop()

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows (2025-12-16 to 2026-01-08)")
st.markdown("---")

report_path = REPORTS_DIR / "feedback_stub.md"
if report_path.exists():
    with st.expander("📄 Feedback Layer Report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))

# --- Time Dynamics ---
st.header("Time Dynamics")
st.markdown("Track how LTV30 evolves across install cohorts.")

cur = get_currency_info()

df["install_date"] = pd.to_datetime(df["install_date"])
daily = df.groupby("install_date").agg(
    users=("vopenid", "count"),
    total_ltv30=("ltv30", "sum"),
    avg_ltv30=("ltv30", "mean"),
).reset_index()
daily["total_ltv30_display"] = convert_vnd(daily["total_ltv30"], cur["code"])
daily["avg_ltv30_display"] = convert_vnd(daily["avg_ltv30"], cur["code"])

fig_time = go.Figure()
fig_time.add_trace(go.Bar(
    x=daily["install_date"], y=daily["total_ltv30_display"],
    name="Total LTV30", marker_color="lightblue", yaxis="y",
))
fig_time.add_trace(go.Scatter(
    x=daily["install_date"], y=daily["avg_ltv30_display"],
    name="Avg LTV30", line=dict(color="red", width=2), yaxis="y2",
))
fig_time.update_layout(
    title="Revenue Dynamics by Install Cohort",
    xaxis_title="Install Date",
    yaxis=dict(title=f"Total LTV30 ({cur['symbol']})", side="left"),
    yaxis2=dict(title=f"Avg LTV30 ({cur['symbol']})", side="right", overlaying="y"),
    height=400,
)
st.plotly_chart(fig_time, use_container_width=True)

# --- Robustness / Stability ---
st.header("Robustness / Stability Check")
st.markdown("Model performance consistency across segments.")

# Generate synthetic per-segment metrics (or real if model available)
if "model" in st.session_state and "X_all" in st.session_state:
    from scipy.stats import spearmanr
    model = st.session_state["model"]
    X_all = st.session_state["X_all"]
    y_all_log = st.session_state["y_all"]
    y_pred_log = model.predict(X_all)
    y_true = np.expm1(y_all_log)
    y_pred = np.expm1(y_pred_log)

    # By country
    country_metrics = []
    for country in df["first_country_code"].unique():
        mask = df["first_country_code"] == country
        if mask.sum() > 50:
            yt = y_true[mask]
            yp = y_pred[mask]
            rho, _ = spearmanr(yt, yp)
            # Lift@10%
            order = np.argsort(-yp)
            sorted_actual = yt[order]
            cumrev = np.cumsum(sorted_actual) / sorted_actual.sum() if sorted_actual.sum() > 0 else np.zeros(len(sorted_actual))
            lift10 = cumrev[int(len(cumrev) * 0.1)] if len(cumrev) > 0 else 0
            country_metrics.append({"Segment": country, "Spearman ρ": round(rho, 3), "Lift@10%": f"{lift10:.1%}", "N": mask.sum()})

    if country_metrics:
        st.subheader("By Country")
        st.dataframe(pd.DataFrame(country_metrics), use_container_width=True)
    st.success("Using live model for stability check.")
else:
    st.info("Using simulated stability metrics. Train a model for real results.")
    # Simulated stability table
    stability_data = [
        {"Segment": "VN", "Spearman ρ": 0.79, "Lift@10%": "76.2%", "AUC": 0.82},
        {"Segment": "TH", "Spearman ρ": 0.83, "Lift@10%": "80.1%", "AUC": 0.85},
        {"Segment": "ID", "Spearman ρ": 0.77, "Lift@10%": "74.8%", "AUC": 0.81},
        {"Segment": "PH", "Spearman ρ": 0.80, "Lift@10%": "78.5%", "AUC": 0.83},
        {"Segment": "MY", "Spearman ρ": 0.78, "Lift@10%": "75.9%", "AUC": 0.82},
    ]
    st.subheader("By Country (Simulated)")
    st.dataframe(pd.DataFrame(stability_data), use_container_width=True)

# --- Install week stability ---
st.subheader("By Install Week")
df["install_week"] = df["install_date"].dt.isocalendar().week.astype(int)
week_stats = df.groupby("install_week").agg(
    users=("vopenid", "count"),
    avg_ltv30=("ltv30", "mean"),
    payer_rate=("is_payer_30", "mean"),
).reset_index()

fig_week = go.Figure()
fig_week.add_trace(go.Bar(
    x=week_stats["install_week"].astype(str), y=week_stats["users"],
    name="Users", marker_color="lightblue", yaxis="y",
))
week_stats["avg_ltv30_display"] = convert_vnd(week_stats["avg_ltv30"], cur["code"])
fig_week.add_trace(go.Scatter(
    x=week_stats["install_week"].astype(str), y=week_stats["avg_ltv30_display"],
    name="Avg LTV30", line=dict(color="red", width=2), yaxis="y2",
))
fig_week.update_layout(
    title="Weekly Cohort Stability",
    xaxis_title="Install Week",
    yaxis=dict(title="Users", side="left"),
    yaxis2=dict(title=f"Avg LTV30 ({cur['symbol']})", side="right", overlaying="y"),
    height=400,
)
st.plotly_chart(fig_week, use_container_width=True)

# --- Planned A/B Tests ---
st.header("Planned A/B Tests")
tests = pd.DataFrame([
    {"Test": "Model vs Random Seeds (FB)", "Hypothesis": "Model seeds yield +20% ROAS", "Status": "🟡 Planned"},
    {"Test": "Top-5% vs Top-10%", "Hypothesis": "Tighter seed = higher precision", "Status": "🟡 Planned"},
    {"Test": "pLTV vs D7-Rev Heuristic", "Hypothesis": "ML model outperforms simple rule", "Status": "🟡 Planned"},
    {"Test": "Country-Specific Models", "Hypothesis": "Local models > global model", "Status": "🔴 Backlog"},
])
st.dataframe(tests, use_container_width=True, hide_index=True)

# --- Feedback Loop Diagram ---
st.header("Feedback Loop")
st.markdown("""
```
Campaign Launch → Seed Export → Ad Network
        ↓
   Install Cohort Observed (D30)
        ↓
   Actual LTV30 Measured
        ↓
   Compare: Model-Seeded vs Control
        ↓
   Update Model / Retrain
```
""")
st.info("This layer closes the Decision-Centric Intelligence Loop by feeding real outcomes back into model retraining.")
