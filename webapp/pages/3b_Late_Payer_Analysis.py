"""
Page 3b — Late Payer Analysis (rev_d7 = 0)
Evaluate incremental ML value inside the rev_d7=0 segment
where heuristic ranking has no power.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, render_top_menu, get_data, get_test_data, convert_vnd, get_currency_info,
    format_currency, REPORTS_DIR, DATA_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking, TEST_DATASETS,
)

render_top_menu()
render_sidebar()

# ── helpers ────────────────────────────────────────────────────────
@st.cache_data
def precompute_cumulative_revenue(y_true, y_scores, n_points=500):
    """Precompute sorted indices and cumulative revenue curve for a strategy."""
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    order = np.argsort(-y_scores)
    sorted_rev = y_true[order]
    total_rev = sorted_rev.sum()
    if total_rev == 0:
        return np.zeros(n_points), np.zeros(n_points), order
    cum_rev = np.cumsum(sorted_rev) / total_rev
    pcts = np.arange(1, len(cum_rev) + 1) / len(cum_rev)
    # Downsample for plotting
    sample_idx = np.linspace(0, len(pcts) - 1, n_points, dtype=int)
    return pcts[sample_idx], cum_rev[sample_idx], order


def revenue_capture_at_k(y_true, order, k_pct):
    """Revenue captured in top K% of users (by pre-sorted order)."""
    k = max(1, int(len(y_true) * k_pct / 100))
    top_rev = y_true[order[:k]].sum()
    total = y_true.sum()
    return top_rev / total if total > 0 else 0.0


def list_available_datasets():
    """List CSV files in the data directory."""
    datasets = {}
    for f in DATA_DIR.glob("cfm_pltv*.csv"):
        size_mb = f.stat().st_size / 1e6
        datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    return datasets


# ── page ───────────────────────────────────────────────────────────
st.title("🔍 Late Payer Analysis (rev_d7 = 0)")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page.")
    st.stop()

cur = get_currency_info()

st.markdown(
    "Users with **rev_d7 = 0** paid nothing in the first 7 days — yet some become high spenders later.  \n"
    "Heuristics like `rev_d7` rank them all equally (zero). This page quantifies how much **incremental value** "
    "the ML model adds by detecting these hidden future payers."
)

# =====================================================================
# DATASET SELECTOR
# =====================================================================
st.header("📂 Select Dataset")
datasets = list_available_datasets()

if not datasets:
    st.error("No datasets found in data/ directory.")
    st.stop()

ds_names = list(datasets.keys())
default_idx = ds_names.index("cfm_pltv") if "cfm_pltv" in ds_names else 0

col_ds1, col_ds2 = st.columns([2, 3])
with col_ds1:
    chosen_ds = st.selectbox(
        "Dataset", ds_names, index=default_idx, key="lpa_dataset",
        help="Choose which dataset to analyze"
    )
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.1f} MB")

# Load dataset
ds_path = ds_info["path"]
df_eval = pd.read_csv(ds_path, low_memory=False)
st.success(f"✅ Loaded **{len(df_eval):,}** rows from {chosen_ds}")

# ── model predictions ─────────────────────────────────────────────
use_live = "model" in st.session_state

if use_live:
    model = st.session_state["model"]
    model_feats = st.session_state.get("model_features", [])
    num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]

    feature_df = df_eval[num_feats + cat_feats].copy()
    for c in cat_feats:
        le = LabelEncoder()
        feature_df[c] = le.fit_transform(feature_df[c].astype(str))
    if "first_charge_day_offset_d7" in feature_df.columns:
        feature_df["first_charge_day_offset_d7"] = feature_df["first_charge_day_offset_d7"].fillna(-1)

    y_pred_all = np.expm1(model.predict(feature_df))
    model_label = f"XGBoost ({len(model_feats)}f)"
else:
    st.info("⬅️ No trained model. Train one on the **Features & Model** page.  \n"
            "Using demo predictions for illustration.")
    rng = np.random.default_rng(42)
    y_pred_all = df_eval["ltv30"].values * rng.uniform(0.6, 1.4, len(df_eval)) + rng.normal(0, 0.5, len(df_eval))
    y_pred_all = np.maximum(y_pred_all, 0)
    model_label = "XGBoost (demo)"

# =====================================================================
# SEGMENT DATA
# =====================================================================
y_true_all = df_eval["ltv30"].values.astype(float)
rev_d7_col = df_eval["rev_d7"].values.astype(float) if "rev_d7" in df_eval.columns else np.zeros(len(df_eval))

mask_d7_zero = rev_d7_col == 0
mask_d7_pos = rev_d7_col > 0

y_true_seg = y_true_all[mask_d7_zero]
y_pred_seg = y_pred_all[mask_d7_zero]
df_seg = df_eval[mask_d7_zero].copy()

n_all = len(y_true_all)
n_zero = mask_d7_zero.sum()
n_pos = mask_d7_pos.sum()
rev_all = y_true_all.sum()
rev_zero = y_true_seg.sum()
rev_pos = y_true_all[mask_d7_pos].sum()

# ── Segment overview KPIs ──────────────────────────────────────────
st.markdown("---")
st.header("📊 Segment Overview")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("D7=0 Users", f"{n_zero:,}", f"{n_zero/n_all:.1%} of total")
with kpi2:
    st.metric("D7=0 Revenue (LTV30)", format_currency(rev_zero, cur["code"]),
              f"{rev_zero/rev_all:.1%} of total" if rev_all > 0 else "0%")
with kpi3:
    payers_in_seg = (y_true_seg > 0).sum()
    st.metric("Late Payers in D7=0", f"{payers_in_seg:,}",
              f"{payers_in_seg/n_zero:.2%} conversion" if n_zero > 0 else "0%")
with kpi4:
    avg_ltv_payers = y_true_seg[y_true_seg > 0].mean() if payers_in_seg > 0 else 0
    st.metric("Avg LTV30 (late payers)", format_currency(avg_ltv_payers, cur["code"]))

st.caption(f"Evaluating on **{chosen_ds}** — **{n_zero:,}** users with rev_d7 = 0 "
           f"(out of {n_all:,} total)")

if n_zero < 100:
    st.error("Too few users in D7=0 segment for meaningful analysis.")
    st.stop()

if rev_zero == 0:
    st.warning("No revenue from D7=0 users in this dataset — all late payer metrics will be zero.")

# =====================================================================
# STRATEGY COMPARISON (within D7=0 segment)
# =====================================================================
st.markdown("---")
st.header("🔀 Strategy Comparison (within D7=0)")
st.markdown(
    "Toggle baselines to compare. Note: **rev_d7 is always zero** in this segment, "
    "so it has **no ranking power** — equivalent to random."
)

cols_bl = st.columns(len(BASELINE_HEURISTICS))
strategies = {model_label: {"scores": y_pred_seg, "color": "royalblue"}}
for i, (name, bl_info) in enumerate(BASELINE_HEURISTICS.items()):
    with cols_bl[i]:
        default_on = name in ("rev_d7 (D7 Revenue)", "active_days_d7 (Active Days)")
        on = st.toggle(name.split(" (")[0], value=default_on, key=f"lpa_bl_{name}",
                        help=bl_info["description"])
        if on:
            bl_scores = compute_baseline_ranking(df_seg, bl_info["column"])
            strategies[name] = {"scores": bl_scores, "color": bl_info["color"]}

# ── Dynamic K selector ────────────────────────────────────────────
st.markdown("---")
k_col1, k_col2 = st.columns([1, 3])
with k_col1:
    k_max = st.slider("Max K% to display", min_value=5, max_value=50, value=10, step=5,
                       key="lpa_k_max", help="Zoom into the top K% of the revenue capture curve")

# =====================================================================
# CHART 1: Revenue Capture Curve (D7=0)
# =====================================================================
color_map = {model_label: "royalblue"}
for name, s_info in strategies.items():
    if name != model_label:
        color_map[name] = s_info["color"]

# Precompute curves
curves = {}
orders = {}
for name, s_info in strategies.items():
    pcts, cum_rev, order = precompute_cumulative_revenue(
        tuple(y_true_seg.tolist()), tuple(s_info["scores"].tolist()), n_points=500
    )
    curves[name] = (pcts, cum_rev)
    orders[name] = order

fig_capture = go.Figure()
for name, (pcts, cum_rev) in curves.items():
    mask_k = pcts <= k_max / 100
    fig_capture.add_trace(go.Scatter(
        x=pcts[mask_k] * 100, y=cum_rev[mask_k] * 100,
        name=name, line=dict(color=color_map.get(name, "blue"), width=2.5),
        hovertemplate="Top %{x:.1f}% users<br>Revenue captured: %{y:.1f}%<extra>" + name + "</extra>",
    ))
# Random baseline
fig_capture.add_trace(go.Scatter(
    x=[0, k_max], y=[0, k_max], name="Random",
    line=dict(dash="dash", color="lightgray", width=1.5),
))
fig_capture.update_layout(
    title=f"Revenue Capture Curve (D7=0 segment, top {k_max}%)",
    xaxis_title="% Users ranked by score",
    yaxis_title="% Cumulative LTV30 Captured",
    height=450, legend=dict(orientation="h", y=-0.18),
    xaxis=dict(range=[0, k_max]), yaxis=dict(range=[0, None]),
)

# =====================================================================
# CHART 2: Revenue Capture @K Table
# =====================================================================
k_values = sorted(set([0.5, 1, 5, k_max]))
table_rows = []
for k_pct in k_values:
    row = {"K%": f"{k_pct}%"}
    model_cap = None
    for name, s_info in strategies.items():
        cap = revenue_capture_at_k(y_true_seg, orders[name], k_pct)
        row[f"{name} Capture"] = f"{cap:.1%}"
        if name == model_label:
            model_cap = cap
    # Incremental lift vs each heuristic
    for name in strategies:
        if name != model_label and model_cap is not None:
            h_cap = revenue_capture_at_k(y_true_seg, orders[name], k_pct)
            lift = model_cap - h_cap
            inc_rev = lift * rev_zero
            row[f"Δ vs {name.split(' (')[0]}"] = f"+{lift:.1%}"
            row[f"Δ Rev vs {name.split(' (')[0]}"] = format_currency(inc_rev, cur["code"])
    table_rows.append(row)

capture_df = pd.DataFrame(table_rows)

# Display side-by-side
col_curve, col_table = st.columns([1.3, 1])
with col_curve:
    st.plotly_chart(fig_capture, use_container_width=True)
with col_table:
    st.subheader("Revenue Capture @K")
    st.dataframe(capture_df, use_container_width=True, hide_index=True, height=420)

# =====================================================================
# CHART 3 & 4: LTV30 Distribution + Decile Breakdown (side-by-side)
# =====================================================================
st.markdown("---")

# --- LTV30 Distribution (D7=0) ---
top1_threshold = np.percentile(y_true_seg, 99) if len(y_true_seg) > 0 else 0
ltv_nonzero = y_true_seg[y_true_seg > 0]
ltv_display = convert_vnd(ltv_nonzero, cur["code"])
top1_display = convert_vnd(top1_threshold, cur["code"])

if len(ltv_display) > 0:
    fig_hist = px.histogram(
        x=ltv_display, nbins=50,
        title=f"LTV30 Distribution — D7=0 Payers Only ({len(ltv_display):,})",
        labels={"x": f"LTV30 ({cur['symbol']})", "y": "Count"},
        color_discrete_sequence=["#FF6600"],
    )
    fig_hist.add_vline(x=top1_display, line_dash="dash", line_color="red",
                       annotation_text=f"Top 1% ≥ {cur['symbol']}{top1_display:,.0f}",
                       annotation_position="top right")
    top1_rev = y_true_seg[y_true_seg >= top1_threshold].sum()
    fig_hist.update_layout(height=420)
else:
    fig_hist = go.Figure()
    fig_hist.update_layout(title="No payers in D7=0 segment", height=420)
    top1_rev = 0

# --- Decile Breakdown (D7=0) ---
model_order = orders[model_label]
n_seg = len(y_true_seg)
decile_labels = []
decile_users = []
decile_avg_ltv = []
decile_rev_share = []

for d in range(10):
    start = int(n_seg * d / 10)
    end = int(n_seg * (d + 1) / 10)
    idx = model_order[start:end]
    rev_d = y_true_seg[idx].sum()
    avg_d = y_true_seg[idx].mean()
    decile_labels.append(f"D{d+1}")
    decile_users.append(len(idx))
    decile_avg_ltv.append(avg_d)
    decile_rev_share.append(rev_d / rev_zero if rev_zero > 0 else 0)

decile_df = pd.DataFrame({
    "Decile": decile_labels,
    "Users": decile_users,
    f"Avg LTV30 ({cur['symbol']})": [format_currency(v, cur["code"]) for v in convert_vnd(np.array(decile_avg_ltv), cur["code"])],
    "Revenue Share": [f"{v:.1%}" for v in decile_rev_share],
})

col_hist, col_decile = st.columns(2)
with col_hist:
    st.subheader("LTV30 Distribution (D7=0 Payers)")
    st.plotly_chart(fig_hist, use_container_width=True)
    if top1_rev > 0:
        st.caption(f"Top 1% of D7=0 payers contribute **{format_currency(top1_rev, cur['code'])}** "
                   f"({top1_rev/rev_zero:.1%} of segment revenue)")
with col_decile:
    st.subheader("Decile Breakdown (by Model Score)")
    st.dataframe(decile_df, use_container_width=True, hide_index=True, height=420)
    st.caption("D1 = highest model score → D10 = lowest. A good model concentrates revenue in D1–D2.")

# =====================================================================
# CHART 5: Whale Discovery
# =====================================================================
st.markdown("---")
st.subheader("🐋 Whale Discovery (D7=0)")
st.markdown(
    "**Whale** = top 1% global revenue contributors. How many whales had rev_d7 = 0, "
    "and can the model still find them?"
)

# Define whales as top 1% by ltv30 globally
whale_threshold = np.percentile(y_true_all, 99)
is_whale_all = y_true_all >= whale_threshold
n_whales_total = is_whale_all.sum()

# Whales within d7_zero segment
is_whale_seg = y_true_seg >= whale_threshold
n_whales_seg = is_whale_seg.sum()

whale_k_values = [1, 5, 10]
whale_rows = []
for k_pct in whale_k_values:
    row = {"K%": f"Top {k_pct}%"}
    for name in strategies:
        k = max(1, int(n_seg * k_pct / 100))
        top_k_idx = orders[name][:k]
        whales_found = is_whale_seg[top_k_idx].sum()
        rate = whales_found / n_whales_seg if n_whales_seg > 0 else 0
        row[name] = f"{whales_found}/{n_whales_seg} ({rate:.0%})"
    whale_rows.append(row)

whale_df = pd.DataFrame(whale_rows)

# Whale bar chart
whale_bar_data = []
for k_pct in whale_k_values:
    for name in strategies:
        k = max(1, int(n_seg * k_pct / 100))
        top_k_idx = orders[name][:k]
        whales_found = is_whale_seg[top_k_idx].sum()
        rate = whales_found / n_whales_seg * 100 if n_whales_seg > 0 else 0
        whale_bar_data.append({"K%": f"Top {k_pct}%", "Strategy": name, "Whales Found (%)": rate})

fig_whale = px.bar(
    pd.DataFrame(whale_bar_data), x="K%", y="Whales Found (%)", color="Strategy",
    barmode="group", title="% of D7=0 Whales Captured in Top K%",
    color_discrete_map=color_map,
)
fig_whale.update_layout(height=400, legend=dict(orientation="h", y=-0.2))

col_whale_chart, col_whale_tbl = st.columns([1.3, 1])
with col_whale_chart:
    st.plotly_chart(fig_whale, use_container_width=True)
with col_whale_tbl:
    st.markdown(f"**{n_whales_seg}** whales (top 1% global) have rev_d7 = 0 "
                f"({n_whales_seg/n_whales_total:.0%} of all whales)" if n_whales_total > 0 else "No whales found")
    st.dataframe(whale_df, use_container_width=True, hide_index=True)

# =====================================================================
# INSIGHT LOGIC
# =====================================================================
st.markdown("---")
st.header("💡 Insights")

# Compute incremental lift at 5% for the main heuristic (rev_d7)
model_cap_5 = revenue_capture_at_k(y_true_seg, orders[model_label], 5)
rev_d7_name = "rev_d7 (D7 Revenue)"
if rev_d7_name in strategies:
    heur_cap_5 = revenue_capture_at_k(y_true_seg, orders[rev_d7_name], 5)
    inc_lift_5 = model_cap_5 - heur_cap_5
    inc_rev_5 = inc_lift_5 * rev_zero

    st.markdown(f"**Model Revenue Capture @5%:** {model_cap_5:.1%}  \n"
                f"**rev_d7 Capture @5%:** {heur_cap_5:.1%}  \n"
                f"**Incremental Lift @5%:** +{inc_lift_5:.1%}  \n"
                f"**Incremental Revenue @5%:** {format_currency(inc_rev_5, cur['code'])}")

    if inc_lift_5 > 0.03:
        st.success("✅ **ML meaningfully improves late payer detection.** "
                   "The model finds hidden revenue that rev_d7 cannot rank.")
    elif inc_lift_5 > 0.01:
        st.info("ℹ️ **ML shows moderate incremental value** over rev_d7 in the D7=0 segment. "
                "Consider the cost-benefit of deploying ML for this segment.")
    else:
        st.warning("⚠️ **ML adds limited incremental value** in the D7=0 segment vs heuristic. "
                   "Behavioral features may not differentiate late payers sufficiently.")
else:
    st.markdown(f"**Model Revenue Capture @5%:** {model_cap_5:.1%}")
    st.info("Toggle **rev_d7** above to see incremental lift comparison.")

# Additional segment-level insight
if payers_in_seg > 0 and rev_zero > 0:
    st.markdown(f"""
**Segment Summary:**
- **{n_zero:,}** users had rev_d7 = 0 ({n_zero/n_all:.1%} of all users)
- **{payers_in_seg:,}** of them became payers by D30 ({payers_in_seg/n_zero:.2%} late conversion rate)
- They generated **{format_currency(rev_zero, cur['code'])}** in LTV30 ({rev_zero/rev_all:.1%} of total revenue)
- The model captures **{model_cap_5:.1%}** of this revenue by targeting just the top 5% of the segment
""")

# ── Validation info (collapsed) ───────────────────────────────────
with st.expander("🔍 Validation Checks", expanded=False):
    cap_100 = revenue_capture_at_k(y_true_seg, orders[model_label], 100)
    st.markdown(f"- Capture @100% = {cap_100:.4f} (should be 1.0000) {'✅' if abs(cap_100 - 1.0) < 0.001 else '❌'}")
    st.markdown(f"- Total D7=0 revenue: {format_currency(rev_zero, cur['code'])}")
    st.markdown(f"- Total dataset revenue: {format_currency(rev_all, cur['code'])}")

    # Check heuristic variance in d7_zero
    if "rev_d7" in df_seg.columns:
        rev_d7_var = df_seg["rev_d7"].var()
        st.markdown(f"- rev_d7 variance in D7=0 segment: {rev_d7_var:.6f} "
                    f"{'✅ (zero — no ranking power)' if rev_d7_var == 0 else '⚠️ unexpected non-zero'}")

    # Monotonicity check
    _, cum_rev_model = curves[model_label]
    is_monotonic = np.all(np.diff(cum_rev_model) >= -1e-10)
    st.markdown(f"- Cumulative capture curve monotonic: {'✅' if is_monotonic else '❌'}")
