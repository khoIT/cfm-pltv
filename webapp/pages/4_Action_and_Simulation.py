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
import os
from shared import (
    render_sidebar, render_top_menu, render_report_md, get_data, format_currency, convert_vnd,
    get_currency_info, REPORTS_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking,
    list_datasets_by_role, load_csv_cached,
)


render_top_menu()
render_sidebar()

st.title("🎮 Action & Simulation")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No dataset selected")
    st.info("Please select a dataset from the **Dataset Registry** in the sidebar.")
    st.stop()

render_report_md(REPORTS_DIR / "action_simulation.md", "📄 Action Simulation Report")

# ── Dataset role selector ─────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Dataset")
role_meta = list_datasets_by_role()

_avail_roles = [r for r in ["train", "test", "recent"] if role_meta.get(r) is not None]
if not _avail_roles:
    _avail_roles = ["train"]

_role_labels = {
    "train":  "🏋️ Train (in-sample)",
    "test":   "🧪 Test (holdout)",
    "recent": "🆕 Recent (live scoring)",
}
_sim_role = st.radio(
    "Simulate on:",
    _avail_roles,
    format_func=lambda r: _role_labels.get(r, r),
    horizontal=True,
    key="sim_role_select",
    help="**Train/Test** = historical users with known LTV30 (strategy comparison).  "
         "**Recent** = users installed <30 days ago — no LTV30 yet, model scores only.",
)

_role_info = role_meta.get(_sim_role)
if _role_info is not None:
    st.caption(
        f"**{_role_info['name']}** — {_role_info['size_mb']:.1f} MB"
        + (f" | {_role_info['split_info']}" if _role_info.get('split_info') else "")
    )
    _path = _role_info["path"]
    _mtime = os.path.getmtime(_path) if os.path.exists(_path) else 0.0
    df_sim = load_csv_cached(_path, _mtime)
else:
    st.warning("Dataset not found — falling back to registry default.")
    df_sim = get_data()

_is_recent_mode = (_sim_role == "recent")
if _is_recent_mode:
    st.info(
        "🆕 **Recent users mode** — LTV30 is not yet realized for these users. "
        "The model will score them, but no ground-truth comparison is possible. "
        "Use this to generate a ranked seed list for your next UA campaign.",
        icon="🆕"
    )
elif _sim_role == "test":
    st.info(
        "🧪 **Test (holdout) mode** — unbiased simulation on users not seen during training.",
        icon="🧪"
    )

st.caption(f"Loaded: **{len(df_sim):,}** rows")

# ── model predictions ─────────────────────────────────────────────
if "model" in st.session_state:
    from sklearn.preprocessing import LabelEncoder
    model = st.session_state["model"]
    model_feats = st.session_state.get("model_features", [])
    num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]

    feature_df_sim = df_sim[[f for f in num_feats + cat_feats if f in df_sim.columns]].copy()
    for c in cat_feats:
        if c in feature_df_sim.columns:
            le = LabelEncoder()
            feature_df_sim[c] = le.fit_transform(feature_df_sim[c].astype(str))
    if "first_charge_day_offset_d7" in feature_df_sim.columns:
        feature_df_sim["first_charge_day_offset_d7"] = feature_df_sim["first_charge_day_offset_d7"].fillna(-1)
    # Align columns to model expectation
    for f in num_feats + cat_feats:
        if f not in feature_df_sim.columns:
            feature_df_sim[f] = 0
    feature_df_sim = feature_df_sim[num_feats + cat_feats]

    y_pred_model = np.expm1(model.predict(feature_df_sim))
    model_name = f"XGBoost ({len(model_feats)}f)"
else:
    rng = np.random.default_rng(42)
    if "ltv30" in df_sim.columns:
        y_pred_model = df_sim["ltv30"].values * rng.uniform(0.6, 1.4, len(df_sim)) + rng.normal(0, 0.5, len(df_sim))
    else:
        y_pred_model = rng.uniform(0, 1000, len(df_sim))
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
n_selected = max(1, int(len(df_sim) * top_k_pct / 100))
model_order = np.argsort(-y_pred_model)
model_cost = n_selected * cpi

# ══════════════════════════════════════════════════════════════════
# RECENT MODE — no ground-truth LTV30; show ranked seed list only
# ══════════════════════════════════════════════════════════════════
if _is_recent_mode:
    st.markdown("---")
    st.header("🆕 Ranked Seed List — Recent Users")
    st.caption(
        f"Top **{top_k_pct}%** = **{n_selected:,}** users selected by predicted LTV30. "
        "LTV30 is not yet realized — these are model scores only."
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Selected Users", f"{n_selected:,}", f"Top {top_k_pct}% of {len(df_sim):,}")
    with kpi2:
        avg_pred = float(y_pred_model[model_order[:n_selected]].mean())
        st.metric(f"Avg Predicted LTV30 ({currency_symbol})",
                  format_currency(convert_vnd(avg_pred, currency), currency))
    with kpi3:
        st.metric(f"Est. Seed Cost ({currency_symbol})",
                  format_currency(convert_vnd(model_cost, currency), currency),
                  f"at CPI {currency_symbol}{format_currency(convert_vnd(cpi, currency), currency)}")

    # Score distribution
    fig_score = go.Figure()
    fig_score.add_trace(go.Histogram(
        x=convert_vnd(y_pred_model, currency),
        nbinsx=50, marker_color="#3498db", opacity=0.7, name="All recent users",
    ))
    threshold_disp = convert_vnd(float(y_pred_model[model_order[n_selected - 1]]), currency)
    fig_score.add_vline(x=threshold_disp, line_dash="dash", line_color="#e74c3c",
                        annotation_text=f"Top {top_k_pct}% cutoff")
    fig_score.update_layout(
        title=f"Predicted LTV30 Distribution — Recent Users ({currency_symbol})",
        xaxis_title=f"Predicted LTV30 ({currency_symbol})",
        yaxis_title="Users", height=380,
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # Ranked seed table
    st.subheader(f"📋 Top {top_k_pct}% Seed List ({n_selected:,} users)")
    seed_df = df_sim.iloc[model_order[:n_selected]].copy()
    seed_df["predicted_ltv30"] = y_pred_model[model_order[:n_selected]]
    seed_df["predicted_ltv30_disp"] = seed_df["predicted_ltv30"].apply(
        lambda v: format_currency(convert_vnd(v, currency), currency))
    seed_df["rank"] = range(1, n_selected + 1)

    display_cols = ["rank", "vopenid"]
    for c in ["install_date", "media_source", "first_os", "rev_d7", "games_d7",
              "active_days_d7", "predicted_ltv30_disp"]:
        if c in seed_df.columns:
            display_cols.append(c)

    st.dataframe(seed_df[display_cols].head(500), use_container_width=True, hide_index=True)
    st.caption("Showing first 500 rows. Download full list:")

    csv_bytes = seed_df[display_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Full Seed List (CSV)",
        data=csv_bytes,
        file_name=f"seed_list_top{top_k_pct}pct_recent.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════
# HISTORICAL MODE — train or test, ltv30 available for comparison
# ══════════════════════════════════════════════════════════════════
strategies["Random"] = {"scores": np.random.default_rng(42).permutation(len(df_sim)).astype(float), "color": "lightgray"}

ltv_vals = df_sim["ltv30"].values if "ltv30" in df_sim.columns else np.zeros(len(df_sim))
total_rev = ltv_vals.sum()

model_top_k_idx = model_order[:n_selected]
model_rev_captured = ltv_vals[model_top_k_idx].sum()
model_pct_captured = model_rev_captured / total_rev * 100 if total_rev > 0 else 0
model_roi = (model_rev_captured - model_cost) / model_cost * 100 if model_cost > 0 else 0

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
k_range = list(range(1, 11))

with chart_col:
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
model_revs_pct = []
for k in k_range:
    n = max(1, int(len(df_sim) * k / 100))
    model_revs_pct.append(ltv_vals[model_order[:n]].sum() / total_rev * 100 if total_rev > 0 else 0)

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
