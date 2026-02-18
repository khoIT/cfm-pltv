"""
Page 4 — Action & Simulation
Top-K selection slider, incremental ROI estimation, uplift and sensitivity charts.
Compare model vs heuristic baselines for seed selection.
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
    render_sidebar, render_top_menu, get_data, get_test_data, format_currency, convert_vnd,
    get_currency_info, REPORTS_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking, TEST_DATASETS,
)

render_top_menu()
render_sidebar()

st.title("🎮 Action & Simulation")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page in the sidebar.")
    st.stop()

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows")

report_path = REPORTS_DIR / "action_simulation.md"
if report_path.exists():
    with st.expander("📄 Action Simulation Report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))

# ═════════════════════════════════════════════════════════════════════
# TEST DATASET SELECTOR
# ═════════════════════════════════════════════════════════════════════
st.header("🧪 Select Test Dataset for Simulation")
st.markdown("Choose which **out-of-time holdout** to simulate seed selection on.")

test_options = list(TEST_DATASETS.keys())
col_s1, col_s2 = st.columns([2, 3])
with col_s1:
    test_choice = st.radio(
        "Test dataset",
        test_options,
        index=0,
        key="action_test_choice",
    )
with col_s2:
    tinfo = TEST_DATASETS[test_choice]
    st.markdown(f"**{test_choice}**")
    st.markdown(f"- 📅 Dates: `{tinfo['dates']}`")
    st.markdown(f"- 📦 Rows: ~{tinfo['rows']}")
    st.markdown(f"- 💡 {tinfo['description']}")

df_sim = get_test_data(test_choice)
st.success(f"✅ Simulating on **{test_choice}** — {len(df_sim):,} rows")

# ── model predictions ─────────────────────────────────────────────
if "model" in st.session_state:
    from sklearn.preprocessing import LabelEncoder
    model = st.session_state["model"]
    model_feats = st.session_state.get("model_features", [])
    num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]

    feature_df_sim = df_sim[num_feats + cat_feats].copy()
    for c in cat_feats:
        le = LabelEncoder()
        feature_df_sim[c] = le.fit_transform(feature_df_sim[c].astype(str))
    if "first_charge_day_offset_d7" in feature_df_sim.columns:
        feature_df_sim["first_charge_day_offset_d7"] = feature_df_sim["first_charge_day_offset_d7"].fillna(-1)

    y_pred_model = np.expm1(model.predict(feature_df_sim))
    model_name = f"XGBoost ({len(model_feats)}f)"
else:
    rng = np.random.default_rng(42)
    y_pred_model = df_sim["ltv30"].values * rng.uniform(0.6, 1.4, len(df_sim)) + rng.normal(0, 0.5, len(df_sim))
    y_pred_model = np.maximum(y_pred_model, 0)
    model_name = "XGBoost (demo)"
    st.info("⬅️ Using demo predictions. Train a model on the **Features & Model** page for real results.")

# ── Simulation Controls ───────────────────────────────────────────
st.header("🎯 UA Budget Simulation — pLTV Impact")

cur = get_currency_info()
currency = cur["code"]
currency_symbol = cur["symbol"]
cpi_default = 10000.0 if currency == "VND" else 0.42
cpi_min = 1000.0 if currency == "VND" else 0.01
cpi_step = 1000.0 if currency == "VND" else 0.05

# ── Controls + Strategy toggles ───────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1.5, 2])
with ctrl_col1:
    top_k_pct = st.slider("Top-K %", 1, 20, 5, 1, help="Realistic UA seed range: 1–10%")
with ctrl_col2:
    cpi = st.number_input(f"CPI ({currency_symbol})", value=cpi_default, min_value=cpi_min, step=cpi_step)
with ctrl_col3:
    # Build strategies dict — toggles in 2 columns
    strategies = {model_name: {"scores": y_pred_model, "color": "#FF6600"}}
    bl_items = list(BASELINE_HEURISTICS.items())
    bl_col1, bl_col2 = st.columns(2)
    for i, (name, info) in enumerate(bl_items):
        with bl_col1 if i % 2 == 0 else bl_col2:
            on = st.toggle(
                name.split(" (")[0],
                value=(name == "rev_d7 (D7 Revenue)"),
                key=f"act_bl_{name}",
                help=info["description"],
            )
            if on:
                strategies[name] = {
                    "scores": compute_baseline_ranking(df_sim, info["column"]),
                    "color": info["color"],
                }

# Always include random as reference
strategies["Random"] = {"scores": np.random.default_rng(42).permutation(len(df_sim)).astype(float), "color": "lightgray"}

# ── Compute results ────────────────────────────────────────────────
n_selected = max(1, int(len(df_sim) * top_k_pct / 100))
ltv_vals = df_sim["ltv30"].values
total_rev = ltv_vals.sum()

# Model results (for KPI cards)
model_order = np.argsort(-y_pred_model)
model_top_k_idx = model_order[:n_selected]
model_rev_captured = ltv_vals[model_top_k_idx].sum()
model_pct_captured = model_rev_captured / total_rev * 100 if total_rev > 0 else 0
model_cost = n_selected * cpi
model_roi = (model_rev_captured - model_cost) / model_cost * 100 if model_cost > 0 else 0

# Random baseline for ROI comparison
random_order = np.random.default_rng(42).permutation(len(df_sim))
random_rev = ltv_vals[random_order[:n_selected]].sum()
random_pct = random_rev / total_rev * 100 if total_rev > 0 else 0
roi_vs_random = model_pct_captured - random_pct

# ── KPI Cards ──────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("🎯 Selected Users", f"{n_selected:,}", help=f"Top {top_k_pct}% of {len(df_sim):,}")
with kpi2:
    st.metric("💰 Predicted Revenue Share", f"{model_pct_captured:.1f}%",
              delta=f"+{roi_vs_random:.1f}% vs random", help=f"Revenue captured by {model_name}")
with kpi3:
    st.metric(f"📈 ROI ({model_name})", f"{model_roi:,.0f}%",
              help=f"(Revenue − Cost) / Cost at CPI={currency_symbol}{cpi:,.0f}")
with kpi4:
    st.metric(f"💵 Revenue ({currency_symbol})", format_currency(model_rev_captured, currency),
              help="Total predicted D30 revenue from selected seeds")

# ── Cumulative LTV Chart + Strategy Table (side-by-side) ──────────
chart_col, table_col = st.columns([1.4, 1])

with chart_col:
    k_range = list(range(1, 11))  # 1–10% realistic range
    
    fig_area = go.Figure()
    for sname, sinfo in strategies.items():
        order = np.argsort(-sinfo["scores"])
        revs_pct = []
        for k in k_range:
            n = max(1, int(len(df_sim) * k / 100))
            rev_k = ltv_vals[order[:n]].sum()
            revs_pct.append(rev_k / total_rev * 100 if total_rev > 0 else 0)
    
        is_model = sname == model_name
        fig_area.add_trace(go.Scatter(
            x=k_range, y=revs_pct, name=sname,
            fill='tozeroy' if is_model else None,
            line=dict(
                color=sinfo["color"], width=3 if is_model else 2,
                dash="solid" if sname != "Random" else "dash",
            ),
            fillcolor="rgba(255,102,0,0.15)" if is_model else None,
        ))
    
    # Add vertical marker at selected Top-K
    fig_area.add_vline(x=top_k_pct, line_dash="dot", line_color="#FF6600", line_width=2,
                       annotation_text=f"Top {top_k_pct}%", annotation_position="top right")
    
    fig_area.update_layout(
        title="Cumulative Revenue Share by Seed Size",
        xaxis_title="Top-K % of Users", yaxis_title="% of Total Revenue Captured",
        xaxis=dict(dtick=1, range=[0.5, 10.5]),
        yaxis=dict(ticksuffix="%"),
        height=500, legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    st.plotly_chart(fig_area, use_container_width=True)

with table_col:
    st.markdown(f"#### 📋 Strategy Comparison @ Top-{top_k_pct}%")
    
    results_rows = []
    for sname, sinfo in strategies.items():
        order = np.argsort(-sinfo["scores"])
        top_k_idx = order[:n_selected]
        rev_captured = ltv_vals[top_k_idx].sum()
        cost = n_selected * cpi
        roi = (rev_captured - cost) / cost * 100 if cost > 0 else 0
        pct_captured = rev_captured / total_rev * 100 if total_rev > 0 else 0
        results_rows.append({
            "Strategy": sname,
            "Revenue Share": f"{pct_captured:.1f}%",
            "ROI": f"{roi:,.0f}%",
        })
    
    res_df = pd.DataFrame(results_rows)
    st.dataframe(res_df, use_container_width=True, hide_index=True, height=420)

# ── Insight Line ───────────────────────────────────────────────────
# Find sweet spot: K% where marginal gain drops below 50% of average
model_revs_pct = []
for k in k_range:
    n = max(1, int(len(df_sim) * k / 100))
    model_revs_pct.append(ltv_vals[model_order[:n]].sum() / total_rev * 100)

sweet_spot_k = top_k_pct
best_marginal = 0
for k_idx in range(1, len(k_range)):
    marginal = model_revs_pct[k_idx] - model_revs_pct[k_idx - 1]
    if k_idx == 1:
        best_marginal = marginal
    elif marginal < best_marginal * 0.4:
        sweet_spot_k = k_range[k_idx - 1]
        break

st.info(
    f"💡 **Top {top_k_pct}% → ≈ {model_pct_captured:.0f}% revenue share** — "
    f"{'seed optimization sweet spot!' if top_k_pct <= sweet_spot_k else f'consider narrowing to Top {sweet_spot_k}% for better ROI.'}"
)

# ── Treatment Sensitivity ─────────────────────────────────────────
st.markdown("---")
st.subheader("Marginal Revenue & ROI by Seed Size")
st.markdown("> When the bars flatten, you've hit diminishing returns.")

k_range_full = list(range(1, 11))
rois, marginals = [], []
prev_rev = 0
for k in k_range_full:
    n = max(1, int(len(df_sim) * k / 100))
    rev = ltv_vals[model_order[:n]].sum()
    cost_k = n * cpi
    rois.append((rev - cost_k) / cost_k * 100 if cost_k > 0 else 0)
    marginals.append(float(convert_vnd(rev - prev_rev, currency)))
    prev_rev = rev

fig_sens = go.Figure()
fig_sens.add_trace(go.Bar(
    x=k_range_full, y=marginals, name=f"Marginal Revenue ({currency_symbol})",
    marker_color="#FF6600", opacity=0.7,
))
fig_sens.add_trace(go.Scatter(
    x=k_range_full, y=rois, name="Cumulative ROI (%)",
    line=dict(color="#E74C3C", width=3), yaxis="y2",
))
fig_sens.update_layout(
    xaxis_title="Top-K (%)", xaxis=dict(dtick=1),
    yaxis=dict(title=f"Marginal Revenue ({currency_symbol})", side="left"),
    yaxis2=dict(title="ROI (%)", side="right", overlaying="y"),
    height=380,
)
st.plotly_chart(fig_sens, width='stretch')

# ── Seed Profile ───────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"👥 Seed Profile (Top-{top_k_pct}% by {model_name})")

model_order_df = df_sim.iloc[model_order].head(n_selected)

col1, col2 = st.columns(2)
with col1:
    fig_ms = px.pie(model_order_df, names="media_source", title="Media Source (Selected Seeds)")
    st.plotly_chart(fig_ms, width='stretch')
with col2:
    fig_cc = px.pie(model_order_df, names="first_country_code", title="Country (Selected Seeds)")
    st.plotly_chart(fig_cc, width='stretch')
