"""
Page 1 — Decision Definition
Describes the business decision, KPIs, and decision blueprint.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, get_data, format_currency, convert_vnd,
    get_currency_info, REPORTS_DIR,
)

render_sidebar()

st.title("📋 Layer 1 — Decision Definition")
st.markdown("---")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page in the sidebar.")
    st.stop()

report_path = REPORTS_DIR / "decision_definition.md"
if report_path.exists():
    with st.expander("📄 Decision Definition Report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows (2025-12-16 to 2026-01-08)")

cur = get_currency_info()

# =====================================================================
# KPI Dashboard
# =====================================================================
st.header("Live KPI Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ARPU (D30)", format_currency(df["ltv30"].mean(), cur["code"]))
with col2:
    st.metric("Paying Rate (D30)", f"{df['is_payer_30'].mean() * 100:.1f}%")
with col3:
    payers = df[df["rev_d7"] > 0]
    multiplier = (payers["ltv30"] / payers["rev_d7"].clip(lower=0.01)).median() if len(payers) > 0 else 0
    st.metric("D7→D30 Multiplier", f"{multiplier:.1f}×")
with col4:
    top1_pct = df.nlargest(max(1, len(df) // 100), "ltv30")["ltv30"].sum() / df["ltv30"].sum() * 100 if df["ltv30"].sum() > 0 else 0
    st.metric("Top-1% Revenue Share", f"{top1_pct:.0f}%")

# =====================================================================
# Cumulative Revenue by Decile
# =====================================================================
st.header("📊 Revenue Concentration by Decile")
st.markdown("> Shows how revenue concentrates in the top deciles. "
            "A steep curve means a **whale-heavy economy** where targeting top users is critical.")

ltv_sorted = df["ltv30"].sort_values(ascending=False).values
total_rev = ltv_sorted.sum()

decile_rows = []
for i in range(10):
    start = int(len(ltv_sorted) * i / 10)
    end = int(len(ltv_sorted) * (i + 1) / 10)
    decile_rev = ltv_sorted[start:end].sum()
    cum_rev = ltv_sorted[:end].sum()
    decile_rows.append({
        "Decile": f"Top {(i+1)*10}%",
        "Users": f"{start:,}–{end:,}",
        f"Decile Revenue ({cur['symbol']})": format_currency(decile_rev, cur["code"]),
        "% of Total": f"{decile_rev/total_rev*100:.1f}%",
        f"Cumulative Revenue ({cur['symbol']})": format_currency(cum_rev, cur["code"]),
        "Cumulative %": f"{cum_rev/total_rev*100:.1f}%",
    })
decile_df = pd.DataFrame(decile_rows)
st.dataframe(decile_df, use_container_width=True, hide_index=True)

# Gini coefficient
n = len(ltv_sorted)
idx = np.arange(1, n + 1)
ltv_asc = np.sort(df["ltv30"].values)
gini = (2 * np.sum(idx * ltv_asc) / (n * np.sum(ltv_asc))) - (n + 1) / n if np.sum(ltv_asc) > 0 else 0
st.metric("Gini Coefficient (Revenue Inequality)", f"{gini:.3f}",
          help="0 = perfectly equal, 1 = one user has all revenue. F2P games typically 0.85–0.95.")

# Lorenz curve
st.subheader("Lorenz Curve — Revenue Inequality")
cum_users = np.linspace(0, 100, 101)
cum_rev_pct = np.array([0] + [ltv_asc[:int(len(ltv_asc)*p/100)].sum() / total_rev * 100 for p in range(1, 101)])

fig_lorenz = go.Figure()
fig_lorenz.add_trace(go.Scatter(x=cum_users, y=cum_rev_pct, name="Actual", line=dict(color="royalblue", width=2)))
fig_lorenz.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Perfect Equality", line=dict(color="gray", dash="dash")))
fig_lorenz.update_layout(
    xaxis_title="% of Users (sorted by LTV30 ascending)",
    yaxis_title="% of Cumulative Revenue",
    height=400, showlegend=True,
)
st.plotly_chart(fig_lorenz, use_container_width=True)

# =====================================================================
# LTV Distribution — converted values
# =====================================================================
st.subheader("LTV30 Distribution")
ltv_nonzero = df[df["ltv30"] > 0].copy()
if len(ltv_nonzero) > 0:
    ltv_nonzero["ltv30_display"] = convert_vnd(ltv_nonzero["ltv30"], cur["code"])
    fig = px.histogram(
        ltv_nonzero, x="ltv30_display", nbins=50,
        title="LTV30 Distribution (Payers Only, Log Scale)",
        log_y=True, labels={"ltv30_display": f"LTV30 ({cur['symbol']})", "count": "Users"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No payers in dataset.")

# =====================================================================
# Paying Rate by Media Source — converted values
# =====================================================================
st.subheader("Paying Rate by Media Source")
media_pay = df.groupby("media_source").agg(
    users=("vopenid", "count"),
    payers=("is_payer_30", "sum"),
    avg_ltv30=("ltv30", "mean"),
).reset_index()
media_pay["pay_rate"] = (media_pay["payers"] / media_pay["users"] * 100).round(1)
media_pay["avg_ltv30_display"] = convert_vnd(media_pay["avg_ltv30"], cur["code"])

fig2 = px.bar(
    media_pay.sort_values("pay_rate", ascending=False),
    x="media_source", y="pay_rate",
    color="avg_ltv30_display", color_continuous_scale="Viridis",
    title="Paying Rate & Avg LTV30 by Media Source",
    labels={"pay_rate": "Paying Rate (%)", "media_source": "Media Source",
            "avg_ltv30_display": f"Avg LTV30 ({cur['symbol']})"},
)
st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# Whale Economy Analysis
# =====================================================================
st.header("🐋 Whale Economy Analysis")
st.markdown("> With high revenue concentration, understanding your whale segments is critical "
            "for UA targeting and retention strategy.")

# Segment users into tiers
df_seg = df.copy()
df_seg["ltv30_display"] = convert_vnd(df_seg["ltv30"], cur["code"])
ltv_q90 = df["ltv30"].quantile(0.90)
ltv_q99 = df["ltv30"].quantile(0.99)

def classify_tier(v):
    if v <= 0: return "Non-Payer"
    if v < ltv_q90: return "Minnow"
    if v < ltv_q99: return "Dolphin"
    return "Whale"

df_seg["tier"] = df_seg["ltv30"].apply(classify_tier)
tier_order = ["Non-Payer", "Minnow", "Dolphin", "Whale"]

tier_stats = df_seg.groupby("tier").agg(
    users=("vopenid", "count"),
    total_rev=("ltv30", "sum"),
    avg_ltv30=("ltv30", "mean"),
    median_ltv30=("ltv30", "median"),
    avg_rev_d7=("rev_d7", "mean"),
    avg_games=("games_d7", "mean"),
    avg_active_days=("active_days_d7", "mean"),
).reindex(tier_order)

tier_stats["% users"] = (tier_stats["users"] / tier_stats["users"].sum() * 100).round(1)
tier_stats["% revenue"] = (tier_stats["total_rev"] / tier_stats["total_rev"].sum() * 100).round(1)

tier_display = pd.DataFrame({
    "Tier": tier_order,
    "Users": tier_stats["users"].values,
    "% Users": tier_stats["% users"].values,
    "% Revenue": tier_stats["% revenue"].values,
    f"Avg LTV30 ({cur['symbol']})": [format_currency(v, cur["code"]) for v in tier_stats["avg_ltv30"].values],
    f"Avg Rev D7 ({cur['symbol']})": [format_currency(v, cur["code"]) for v in tier_stats["avg_rev_d7"].values],
    "Avg Games D7": tier_stats["avg_games"].values.round(1),
    "Avg Active Days": tier_stats["avg_active_days"].values.round(1),
})
st.dataframe(tier_display, use_container_width=True, hide_index=True)

# Tier revenue pie
col_pie1, col_pie2 = st.columns(2)
with col_pie1:
    fig_pie = px.pie(
        names=tier_order,
        values=tier_stats["total_rev"].values,
        title="Revenue Share by Tier",
        color_discrete_sequence=["#bdc3c7", "#3498db", "#e67e22", "#e74c3c"],
    )
    st.plotly_chart(fig_pie, use_container_width=True)
with col_pie2:
    fig_pie2 = px.pie(
        names=tier_order,
        values=tier_stats["users"].values,
        title="User Count by Tier",
        color_discrete_sequence=["#bdc3c7", "#3498db", "#e67e22", "#e74c3c"],
    )
    st.plotly_chart(fig_pie2, use_container_width=True)

# Whale media source distribution
st.subheader("🎯 Where Do Whales Come From?")
st.markdown("> Knowing which media sources produce whales helps **optimize UA spend** for maximum ROAS.")
whales = df_seg[df_seg["tier"] == "Whale"]
if len(whales) > 0:
    whale_source = whales.groupby("media_source").agg(
        whale_count=("vopenid", "count"),
        whale_rev=("ltv30", "sum"),
    ).reset_index()
    total_users_by_source = df_seg.groupby("media_source")["vopenid"].count().reset_index()
    total_users_by_source.columns = ["media_source", "total_users"]
    whale_source = whale_source.merge(total_users_by_source, on="media_source")
    whale_source["whale_rate"] = (whale_source["whale_count"] / whale_source["total_users"] * 100).round(2)
    whale_source["avg_whale_ltv"] = whale_source["whale_rev"] / whale_source["whale_count"]
    whale_source["avg_whale_ltv_display"] = convert_vnd(whale_source["avg_whale_ltv"], cur["code"])
    whale_source = whale_source.sort_values("whale_count", ascending=False)

    fig_ws = px.bar(
        whale_source, x="media_source", y="whale_count",
        color="whale_rate",
        color_continuous_scale="Reds",
        title="Whale Count & Whale Rate by Media Source",
        labels={"whale_count": "Whale Count", "media_source": "Media Source", "whale_rate": "Whale Rate (%)"},
    )
    st.plotly_chart(fig_ws, use_container_width=True)

# Whale early signals
st.subheader("🔮 Early Signals: D7 Behavior of Whales vs Others")
st.markdown("> If whales show distinctive D7 behavior, we can **predict them early** and act fast.")

signal_cols = ["rev_d7", "txn_cnt_d7", "games_d7", "active_days_d7", "login_rows_d7"]
available_signals = [c for c in signal_cols if c in df_seg.columns]
signal_data = df_seg.groupby("tier")[available_signals].mean().reindex(tier_order)
signal_data_display = signal_data.copy()
for c in ["rev_d7"]:
    if c in signal_data_display.columns:
        signal_data_display[c] = convert_vnd(signal_data_display[c], cur["code"])

fig_signals = go.Figure()
for col in available_signals:
    vals = signal_data[col].values
    # Normalize for display
    max_val = vals.max() if vals.max() > 0 else 1
    fig_signals.add_trace(go.Bar(
        name=col, x=tier_order, y=vals / max_val * 100,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
fig_signals.update_layout(
    title="D7 Feature Means by Tier (Normalized %)",
    barmode="group", height=400,
    yaxis_title="% of Max Value",
)
st.plotly_chart(fig_signals, use_container_width=True)
