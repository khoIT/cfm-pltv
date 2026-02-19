"""
Page 6 — pLTV 30d Analysis Summary & Next Steps
Summarises all analytical activities, model performance, business impacts, and recommended actions.
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
    render_sidebar, render_top_menu, render_report_md,
    get_currency_info, format_currency, convert_vnd,
    REPORTS_DIR,
)

render_top_menu()
render_sidebar()

st.title("📋 pLTV 30d Analysis — Summary & Next Steps")
st.markdown(
    "A consolidated view of all analytical work, model performance, business impacts, "
    "and recommended actions for the CrossFire Mobile pLTV prediction system."
)

# ── Full report expander ───────────────────────────────────────────
render_report_md(
    REPORTS_DIR / "pLTV_Summary.md",
    expander_label="📄 Full Summary Report",
    expanded=True,
)

# ── Quick-reference KPI cards ──────────────────────────────────────
st.markdown("---")
st.header("🏆 Key Results at a Glance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Training Users", "~870k", help="Dec 16 2025 – Jan 8 2026")
with col2:
    st.metric("Spearman ρ (Test 1)", "~0.85+", help="XGBoost vs rev_d7 baseline ~0.75")
with col3:
    st.metric("Lift@10%", "~55–65%", help="Top 10% users capture 55–65% of revenue")
with col4:
    st.metric("Late Payer Rate", "~3%", help="Users who pay only after D7 — core ML opportunity")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric("D7/D30 Revenue Ratio", "~39%", help="D7 revenue is only 39% of D30 — late payers matter")
with col6:
    st.metric("ARPU Spread (channels)", "2.7×", help="Apple Search Ads vs Google Ads ARPU gap")
with col7:
    st.metric("D3 Accuracy Retention", "~97%", help="D3 model retains 97% of D7 Spearman ρ")
with col8:
    st.metric("Est. ROAS Improvement", "+20–40%", help="Combined initiatives vs D7-only baseline")

# ── Priority action matrix ─────────────────────────────────────────
st.markdown("---")
st.header("🚀 Priority Action Matrix")

actions = pd.DataFrame([
    {"#": 1, "Action": "Deploy model-ranked seed lists to all networks", "Timeline": "Week 1–2", "Impact": "High", "Effort": "Low", "Confidence": "High"},
    {"#": 2, "Action": "A/B test enriched seed vs D7-only seed", "Timeline": "Week 1–2", "Impact": "High", "Effort": "Medium", "Confidence": "High"},
    {"#": 3, "Action": "Weekly Spearman ρ drift monitoring", "Timeline": "Week 1–2", "Impact": "Medium", "Effort": "Low", "Confidence": "High"},
    {"#": 4, "Action": "Build D3 feature aggregation SQL pipeline", "Timeline": "Month 1", "Impact": "High", "Effort": "Medium", "Confidence": "Medium"},
    {"#": 5, "Action": "Deploy D3 scoring for faster bid optimisation", "Timeline": "Month 1", "Impact": "High", "Effort": "High", "Confidence": "Medium"},
    {"#": 6, "Action": "Channel-specific seed lists per media source", "Timeline": "Month 1", "Impact": "Medium", "Effort": "Low", "Confidence": "High"},
    {"#": 7, "Action": "Design engagement A/B test (D7 non-payers)", "Timeline": "Month 1", "Impact": "Medium", "Effort": "Medium", "Confidence": "Low"},
    {"#": 8, "Action": "Monthly model retraining cadence", "Timeline": "Month 2–3", "Impact": "Medium", "Effort": "Low", "Confidence": "High"},
    {"#": 9, "Action": "Multi-window ensemble (D1+D3+D7)", "Timeline": "Month 2–3", "Impact": "Medium", "Effort": "High", "Confidence": "Medium"},
    {"#": 10, "Action": "Real-time scoring API (<24h from install)", "Timeline": "Q2+", "Impact": "High", "Effort": "Very High", "Confidence": "Medium"},
])

# Color-code by impact
def highlight_impact(row):
    color = {"High": "background-color: #d4edda", "Medium": "background-color: #fff3cd", "Low": "background-color: #f8d7da"}.get(row["Impact"], "")
    return [color] * len(row)

st.dataframe(
    actions.style.apply(highlight_impact, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── Business impact chart ──────────────────────────────────────────
st.markdown("---")
st.header("💰 Estimated Business Impact by Initiative")

impact_data = pd.DataFrame([
    {"Initiative": "Model seeds vs D7-only", "ROAS Uplift (%)": 15, "Confidence": "High"},
    {"Initiative": "Enriched seeds (+ late payers)", "ROAS Uplift (%)": 7, "Confidence": "High"},
    {"Initiative": "Channel budget reallocation", "ROAS Uplift (%)": 20, "Confidence": "Medium"},
    {"Initiative": "D3 faster optimisation", "ROAS Uplift (%)": 5, "Confidence": "Medium"},
    {"Initiative": "Engagement nudges (A/B)", "ROAS Uplift (%)": 2, "Confidence": "Low"},
])

color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
fig = px.bar(
    impact_data, x="Initiative", y="ROAS Uplift (%)",
    color="Confidence",
    color_discrete_map=color_map,
    title="Estimated ROAS Uplift by Initiative (% improvement vs D7-only baseline)",
    text="ROAS Uplift (%)",
)
fig.update_traces(texttemplate="+%{text}%", textposition="outside")
fig.update_layout(height=420, xaxis_tickangle=-20, legend_title="Confidence Level",
                  yaxis_title="Estimated ROAS Uplift (%)")
st.plotly_chart(fig, use_container_width=True)

st.info(
    "💡 **Note:** Impact estimates are based on industry benchmarks and observational analysis. "
    "A/B tests are required to confirm causal effects. Confidence levels reflect data support strength."
)

# ── Whale Intelligence Overview ────────────────────────────────────
st.markdown("---")
st.header("🐋 Whale Intelligence — AI-Generated Analysis")
st.markdown(
    "A new suite of AI-generated analyses explores the **extreme revenue concentration** "
    "in CrossFire Vietnam. Top 1% of users drive ~79% of revenue — these analyses focus on "
    "identifying, retaining, and expanding the whale segment."
)

render_report_md(
    REPORTS_DIR / "Whale_Analysis_Overview.md",
    expander_label="📄 Whale Analysis Overview Report",
    expanded=False,
)

whale_actions = pd.DataFrame([
    {"Analysis": "🐋 Whale Segmentation", "Key Finding": "Top 1% = 79% of revenue; whales play 3.3× more games", "Action": "VIP onboarding within D1; whale-only UA seeds"},
    {"Analysis": "⏱️ Time-to-First-Purchase", "Key Finding": "35% of payers convert D0; 78% by D3", "Action": "In-session offer trigger; D3 push for non-converters"},
    {"Analysis": "📡 Channel × Whale Quality", "Key Finding": "Whale rate varies significantly by channel", "Action": "Report whale rate in dashboards; reallocate budget"},
    {"Analysis": "📉 Churn Prediction", "Key Finding": "txn_cnt_d7 ≥ 2 is strongest retention signal", "Action": "Retention offer for churn_score > 0.7 payers"},
    {"Analysis": "🎯 Skill-to-Spend", "Key Finding": "High-skill non-payers = best conversion targets", "Action": "Competitive/prestige offers for top-quartile K/D non-payers"},
])
st.dataframe(whale_actions, use_container_width=True, hide_index=True)

col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
with col_l1:
    st.page_link("pages/3h_Whale_Segmentation.py", label="🐋 Whale Segmentation")
with col_l2:
    st.page_link("pages/3i_Time_to_First_Purchase.py", label="⏱️ Time-to-Purchase")
with col_l3:
    st.page_link("pages/3j_Channel_Whale_Quality.py", label="📡 Channel Quality")
with col_l4:
    st.page_link("pages/3k_Churn_Prediction.py", label="📉 Churn Prediction")
with col_l5:
    st.page_link("pages/3l_Skill_Spend_Correlation.py", label="🎯 Skill-to-Spend")
