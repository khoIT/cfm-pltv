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
    render_sidebar, get_data, get_test_data, format_currency, convert_vnd,
    get_currency_info, REPORTS_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking, TEST_DATASETS,
)

render_sidebar()

st.title("🎮 Layer 4 — Action & Simulation")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page in the sidebar.")
    st.stop()

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows")
st.markdown("---")

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
st.header("🎯 Top-K Seed Selection Simulator")
st.markdown(
    "Choose the **Top-K %** of users to include in your UA seed list.  \n"
    "The simulator estimates **revenue captured** and **ROI** for each ranking strategy."
)

cur = get_currency_info()
currency = cur["code"]
currency_symbol = cur["symbol"]
cpi_default = 10000.0 if currency == "VND" else 0.42
cpi_min = 1000.0 if currency == "VND" else 0.01
cpi_step = 1000.0 if currency == "VND" else 0.05

col1, col2 = st.columns(2)
with col1:
    top_k_pct = st.slider("Top-K % of users to select", 1, 50, 10, 1)
with col2:
    cpi = st.number_input(f"Assumed CPI ({currency_symbol})", value=cpi_default, min_value=cpi_min, step=cpi_step)

# ── Strategy selector (model + baselines) ─────────────────────────
st.markdown("---")
st.subheader("🔀 Select Strategies to Compare")
st.markdown("Toggle strategies ON/OFF. Each row below shows what happens if you use that strategy to pick your Top-K seeds.")

# Build strategies dict: name -> (scores, color)
strategies = {model_name: {"scores": y_pred_model, "color": "royalblue"}}

cols = st.columns(len(BASELINE_HEURISTICS))
for i, (name, info) in enumerate(BASELINE_HEURISTICS.items()):
    with cols[i]:
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

# Always include random
strategies["Random"] = {"scores": np.random.default_rng(42).permutation(len(df_sim)).astype(float), "color": "lightgray"}

# ── Compute Top-K results for each strategy ────────────────────────
n_selected = max(1, int(len(df_sim) * top_k_pct / 100))
ltv_vals = df_sim["ltv30"].values
total_rev = ltv_vals.sum()

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
        "Users": n_selected,
        f"Revenue Captured ({currency_symbol})": round(rev_captured if currency == "VND" else rev_captured / 24000, 2 if currency == "USD" else 0),
        "% of Total Revenue": f"{pct_captured:.1f}%",
        f"Cost ({currency_symbol})": round(cost if currency == "VND" else cost / 24000, 2 if currency == "USD" else 0),
        "ROI (%)": round(roi, 0),
    })

# ── Results Table ──────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📋 Top-{top_k_pct}% Seed Selection — Strategy Comparison")
res_df = pd.DataFrame(results_rows)
st.dataframe(res_df, width='stretch', hide_index=True)

# Highlight winner
if len(res_df) > 1:
    rev_col = f"Revenue Captured ({currency_symbol})"
    best = res_df.loc[res_df[rev_col].idxmax(), "Strategy"]
    worst = res_df.loc[res_df[rev_col].idxmin(), "Strategy"]
    best_rev = res_df.loc[res_df[rev_col].idxmax(), rev_col]
    worst_rev = res_df.loc[res_df[rev_col].idxmin(), rev_col]
    delta = best_rev - worst_rev
    delta_str = f"{currency_symbol}{delta:,.2f}" if currency == "USD" else f"{currency_symbol}{delta:,.0f}"
    st.markdown(f"**Best:** `{best}` captures **{delta_str} more** revenue than `{worst}` at Top-{top_k_pct}%.")

# ── Uplift Curve — all strategies ──────────────────────────────────
st.markdown("---")
st.subheader("Uplift Curve: Revenue by Top-K %")
st.markdown("> Cumulative actual revenue captured as you expand the seed list from 1% to 50% for each strategy.")

k_range = list(range(1, 51))
fig_uplift = go.Figure()

for sname, sinfo in strategies.items():
    order = np.argsort(-sinfo["scores"])
    revs = []
    for k in k_range:
        n = max(1, int(len(df_sim) * k / 100))
        revs.append(float(convert_vnd(ltv_vals[order[:n]].sum(), currency)))
    fig_uplift.add_trace(go.Scatter(
        x=k_range, y=revs, name=sname,
        line=dict(color=sinfo["color"], width=2, dash="dash" if sname == "Random" else "solid"),
    ))

fig_uplift.update_layout(
    xaxis_title="Top-K (%)", yaxis_title=f"Cumulative Revenue ({currency_symbol})",
    height=420, legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig_uplift, width='stretch')

# ── Treatment Sensitivity (model only) ─────────────────────────────
st.subheader("Treatment Sensitivity: Marginal Revenue & ROI")
st.markdown("> How does each additional 1% of users contribute to revenue? "
            "Steep drop = diminishing returns = you've found the sweet spot.")

model_order = np.argsort(-y_pred_model)
rois, marginals = [], []
prev_rev = 0
for k in k_range:
    n = max(1, int(len(df_sim) * k / 100))
    rev = ltv_vals[model_order[:n]].sum()
    cost_k = n * cpi
    rois.append((rev - cost_k) / cost_k * 100 if cost_k > 0 else 0)
    marginals.append(float(convert_vnd(rev - prev_rev, currency)))
    prev_rev = rev

fig_sens = go.Figure()
fig_sens.add_trace(go.Bar(
    x=k_range, y=marginals, name=f"Marginal Revenue ({currency_symbol})", marker_color="lightblue",
))
fig_sens.add_trace(go.Scatter(
    x=k_range, y=rois, name="Cumulative ROI (%)", line=dict(color="red", width=2), yaxis="y2",
))
fig_sens.update_layout(
    xaxis_title="Top-K (%)",
    yaxis=dict(title=f"Marginal Revenue ({currency_symbol})", side="left"),
    yaxis2=dict(title="ROI (%)", side="right", overlaying="y"),
    height=400,
)
st.plotly_chart(fig_sens, width='stretch')

# ── Seed Profile ───────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"👥 Selected Seed Profile (Top-{top_k_pct}% by {model_name})")

model_order_df = df_sim.iloc[np.argsort(-y_pred_model)].head(n_selected)

col1, col2 = st.columns(2)
with col1:
    fig_ms = px.pie(model_order_df, names="media_source", title="Media Source (Selected Seeds)")
    st.plotly_chart(fig_ms, width='stretch')
with col2:
    fig_cc = px.pie(model_order_df, names="first_country_code", title="Country (Selected Seeds)")
    st.plotly_chart(fig_cc, width='stretch')

# ── Educational note ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 How to Use This")
st.markdown(
    "1. **Pick your Top-K %** — start at 5–10% for concentrated, high-quality seeds.  \n"
    "2. **Compare strategies** — if `rev_d7` alone is 90% as good as XGBoost, the model adds marginal value.  \n"
    "3. **Check the marginal revenue chart** — when the bars flatten, you've hit diminishing returns.  \n"
    "4. **Export the seed list** — feed the Top-K user IDs to your ad network for lookalike targeting."
)
