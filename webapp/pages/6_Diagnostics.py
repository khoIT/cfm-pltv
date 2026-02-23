"""
Page 6 — Diagnostics & Stability
Compare lift curves across Test 1 vs Test 2 to check model stability across time periods.
Detect revenue concentration risk and model degradation signals.
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
    render_sidebar, render_top_menu, get_data, convert_vnd,
    get_currency_info, format_currency, REPORTS_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking,
    ALL_NUMERIC_FEATURES, ALL_CAT_FEATURES,
)

render_top_menu()
render_sidebar()

st.title("🔬 Diagnostics")
st.markdown("---")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No dataset selected")
    st.info("Please select a dataset from the **Dataset Registry** in the sidebar.")
    st.stop()

cur = get_currency_info()

# =====================================================================
# Load dataset from registry and split by date for temporal comparison
# =====================================================================
st.header("📊 Side-by-Side: Early vs Late Cohorts")
st.markdown(
    "Comparing model performance across two **temporal halves** of the dataset reveals "
    "whether the model generalizes or degrades over time."
)

df_full = get_data()
if "install_date" not in df_full.columns:
    st.error("Dataset must contain 'install_date' column for temporal diagnostics.")
    st.stop()

df_full["install_date"] = pd.to_datetime(df_full["install_date"], errors="coerce")
median_date = df_full["install_date"].median()
df_t1 = df_full[df_full["install_date"] <= median_date].copy()
df_t2 = df_full[df_full["install_date"] > median_date].copy()

if len(df_t1) == 0 or len(df_t2) == 0:
    st.error("Dataset too small to split into two temporal halves.")
    st.stop()

t1_min, t1_max = df_t1["install_date"].min().date(), df_t1["install_date"].max().date()
t2_min, t2_max = df_t2["install_date"].min().date(), df_t2["install_date"].max().date()

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.metric("Early Cohort", f"{len(df_t1):,} rows")
    st.caption(f"📅 {t1_min} to {t1_max}")
with col_info2:
    st.metric("Late Cohort", f"{len(df_t2):,} rows")
    st.caption(f"📅 {t2_min} to {t2_max}")

# =====================================================================
# Helper: compute lift curve
# =====================================================================
def compute_lift(y_true, y_pred):
    """Compute lift curve: cumulative % of revenue captured vs % of users."""
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    cum_rev = np.cumsum(y_sorted) / y_sorted.sum()
    pcts = np.arange(1, len(cum_rev) + 1) / len(cum_rev)
    return pcts, cum_rev


def compute_spearman(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def prepare_features(df_test, model_feats):
    """Prepare test features for model prediction."""
    num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    feature_df = df_test[num_feats + cat_feats].copy()
    for c in cat_feats:
        le = LabelEncoder()
        feature_df[c] = le.fit_transform(feature_df[c].astype(str))
    if "first_charge_day_offset_d7" in feature_df.columns:
        feature_df["first_charge_day_offset_d7"] = feature_df["first_charge_day_offset_d7"].fillna(-1)
    return feature_df


# =====================================================================
# Lift Curve Comparison
# =====================================================================
st.markdown("---")
st.header("📈 Lift Curve: Test 1 vs Test 2")

use_live = "model" in st.session_state
strategies = {}

if use_live:
    model = st.session_state["model"]
    model_feats = st.session_state.get("model_features", [])
    model_label = f"XGBoost ({len(model_feats)}f)"

    X_t1 = prepare_features(df_t1, model_feats)
    X_t2 = prepare_features(df_t2, model_feats)
    pred_t1 = np.expm1(model.predict(X_t1))
    pred_t2 = np.expm1(model.predict(X_t2))

    strategies[f"{model_label} — Early"] = {"y_true": df_t1["ltv30"].values, "y_pred": pred_t1, "color": "royalblue", "dash": "solid"}
    strategies[f"{model_label} — Late"] = {"y_true": df_t2["ltv30"].values, "y_pred": pred_t2, "color": "royalblue", "dash": "dash"}
else:
    st.info("⬅️ Train a model on the **Features & Model** page to see XGBoost lift curves.  \n"
            "Baseline heuristics are shown below.")

# Add baselines
for bl_name, bl_info in BASELINE_HEURISTICS.items():
    col = bl_info["column"]
    if col in df_t1.columns and col in df_t2.columns:
        strategies[f"{bl_name} — Early"] = {
            "y_true": df_t1["ltv30"].values,
            "y_pred": compute_baseline_ranking(df_t1, col),
            "color": bl_info["color"], "dash": "solid",
        }
        strategies[f"{bl_name} — Late"] = {
            "y_true": df_t2["ltv30"].values,
            "y_pred": compute_baseline_ranking(df_t2, col),
            "color": bl_info["color"], "dash": "dash",
        }

# Plot lift curves
fig_lift = go.Figure()
for name, s in strategies.items():
    pcts, cum_rev = compute_lift(s["y_true"], s["y_pred"])
    sample_idx = np.linspace(0, len(pcts) - 1, 200, dtype=int)
    fig_lift.add_trace(go.Scatter(
        x=pcts[sample_idx] * 100, y=cum_rev[sample_idx] * 100,
        name=name,
        line=dict(color=s["color"], width=2, dash=s["dash"]),
    ))
fig_lift.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], name="Random", line=dict(dash="dot", color="lightgray"),
))
fig_lift.update_layout(
    xaxis_title="% Users (ranked by strategy)",
    yaxis_title="% Cumulative Revenue Captured",
    height=500, legend=dict(orientation="h", y=-0.2),
    title="Lift Curve Comparison: Early (solid) vs Late (dashed)",
)
st.plotly_chart(fig_lift, width='stretch')

st.markdown(
    "> **Solid lines** = Early cohort (nearer to start).  \n"
    "> **Dashed lines** = Late cohort (further from start).  \n"
    "> If dashed lines are significantly lower, the model may be **degrading over time**."
)

# =====================================================================
# Metrics Comparison Table
# =====================================================================
st.markdown("---")
st.header("📋 Metrics Comparison: Early vs Late Cohort")

metric_rows = []

def compute_metrics_for(y_true, y_pred, label):
    rho = compute_spearman(y_true, y_pred)
    pcts, cum_rev = compute_lift(y_true, y_pred)
    lift_10 = cum_rev[int(len(cum_rev) * 0.1)] if len(cum_rev) > 0 else 0
    lift_5 = cum_rev[int(len(cum_rev) * 0.05)] if len(cum_rev) > 0 else 0
    return {
        "Strategy": label,
        "Spearman ρ": round(rho, 4),
        "Lift@5%": f"{lift_5:.1%}",
        "Lift@10%": f"{lift_10:.1%}",
        f"Avg Actual LTV30 ({cur['symbol']})": format_currency(y_true.mean(), cur["code"]),
        "Total Users": f"{len(y_true):,}",
    }


if use_live:
    metric_rows.append(compute_metrics_for(df_t1["ltv30"].values, pred_t1, f"{model_label} — Early"))
    metric_rows.append(compute_metrics_for(df_t2["ltv30"].values, pred_t2, f"{model_label} — Late"))

# Add one baseline for comparison
if "rev_d7" in df_t1.columns:
    metric_rows.append(compute_metrics_for(df_t1["ltv30"].values, df_t1["rev_d7"].fillna(0).values, "rev_d7 — Early"))
    metric_rows.append(compute_metrics_for(df_t2["ltv30"].values, df_t2["rev_d7"].fillna(0).values, "rev_d7 — Late"))

if metric_rows:
    st.dataframe(pd.DataFrame(metric_rows), width='stretch', hide_index=True)

    # Highlight degradation
    if use_live and len(metric_rows) >= 2:
        rho_t1 = metric_rows[0]["Spearman ρ"]
        rho_t2 = metric_rows[1]["Spearman ρ"]
        delta = rho_t2 - rho_t1
        if delta < -0.05:
            st.warning(f"⚠️ **Degradation detected:** Spearman ρ dropped by {abs(delta):.4f} from Test 1 to Test 2.  \n"
                       f"The model may be losing accuracy on more recent data.")
        elif delta > 0.05:
            st.success(f"✅ Model improved on Test 2 (+{delta:.4f} Spearman ρ). Stable or improving.")
        else:
            st.success(f"✅ Model is stable across periods (Δρ = {delta:+.4f}).")

# =====================================================================
# Revenue Distribution Comparison
# =====================================================================
st.markdown("---")
st.header("💰 Revenue Distribution: Early vs Late Cohort")
st.markdown("> If the revenue distribution shifts significantly between periods, model accuracy may vary.")

# Ensure is_payer_30 exists
if "is_payer_30" not in df_t1.columns:
    df_t1["is_payer_30"] = (df_t1["ltv30"] > 0).astype(int)
    df_t2["is_payer_30"] = (df_t2["ltv30"] > 0).astype(int)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Early Cohort")
    st.metric("Mean LTV30", format_currency(df_t1["ltv30"].mean(), cur["code"]))
    st.metric("Median LTV30", format_currency(df_t1["ltv30"].median(), cur["code"]))
    st.metric("Payer Rate", f"{df_t1['is_payer_30'].mean() * 100:.1f}%")
    st.metric("Total Revenue", format_currency(df_t1["ltv30"].sum(), cur["code"]))

with col2:
    st.subheader("Late Cohort")
    st.metric("Mean LTV30", format_currency(df_t2["ltv30"].mean(), cur["code"]))
    st.metric("Median LTV30", format_currency(df_t2["ltv30"].median(), cur["code"]))
    st.metric("Payer Rate", f"{df_t2['is_payer_30'].mean() * 100:.1f}%")
    st.metric("Total Revenue", format_currency(df_t2["ltv30"].sum(), cur["code"]))

# Overlaid histograms
t1_payers = df_t1[df_t1["ltv30"] > 0].copy()
t2_payers = df_t2[df_t2["ltv30"] > 0].copy()
t1_payers["ltv30_display"] = convert_vnd(t1_payers["ltv30"], cur["code"])
t2_payers["ltv30_display"] = convert_vnd(t2_payers["ltv30"], cur["code"])
t1_payers["Period"] = f"Early ({t1_min} to {t1_max})"
t2_payers["Period"] = f"Late ({t2_min} to {t2_max})"
combined = pd.concat([t1_payers[["ltv30_display", "Period"]], t2_payers[["ltv30_display", "Period"]]])

fig_hist = px.histogram(
    combined, x="ltv30_display", color="Period",
    nbins=50, barmode="overlay", opacity=0.6,
    title="LTV30 Distribution (Payers Only) — Early vs Late Cohort",
    labels={"ltv30_display": f"LTV30 ({cur['symbol']})", "count": "Users"},
    log_y=True,
    height=420,
)

# ── Media Source Stability (compute early for side-by-side) ──────────
ms_t1 = df_t1.groupby("media_source").agg(
    users=("vopenid", "count"),
    avg_ltv=("ltv30", "mean"),
    payer_rate=("is_payer_30", "mean"),
).reset_index()
ms_t1["period"] = "Early"
ms_t1["avg_ltv_display"] = convert_vnd(ms_t1["avg_ltv"], cur["code"])

ms_t2 = df_t2.groupby("media_source").agg(
    users=("vopenid", "count"),
    avg_ltv=("ltv30", "mean"),
    payer_rate=("is_payer_30", "mean"),
).reset_index()
ms_t2["period"] = "Late"
ms_t2["avg_ltv_display"] = convert_vnd(ms_t2["avg_ltv"], cur["code"])

ms_combined = pd.concat([ms_t1, ms_t2])
fig_ms = px.bar(
    ms_combined, x="media_source", y="avg_ltv_display", color="period",
    barmode="group",
    title=f"Avg LTV30 by Media Source: Early vs Late Cohort",
    labels={"avg_ltv_display": f"Avg LTV30 ({cur['symbol']})", "media_source": "Media Source"},
    height=420,
)

# Display side-by-side
col_hist, col_ms = st.columns(2)
with col_hist:
    st.plotly_chart(fig_hist, use_container_width=True)
with col_ms:
    st.plotly_chart(fig_ms, use_container_width=True)

# =====================================================================
# Concentration Risk Analysis
# =====================================================================
st.markdown("---")
st.header("⚠️ Concentration Risk Analysis")
st.markdown(
    "> How dependent is each test period on **top whales**? "
    "High concentration = higher variance in model-based seed performance."
)

for label, df_test in [("Early Cohort", df_t1), ("Late Cohort", df_t2)]:
    ltv = df_test["ltv30"].values
    total = ltv.sum()
    sorted_ltv = np.sort(ltv)[::-1]
    n = len(sorted_ltv)

    top_1 = sorted_ltv[:max(1, n // 100)].sum() / total * 100 if total > 0 else 0
    top_5 = sorted_ltv[:max(1, n // 20)].sum() / total * 100 if total > 0 else 0
    top_10 = sorted_ltv[:max(1, n // 10)].sum() / total * 100 if total > 0 else 0

    # Gini
    idx = np.arange(1, n + 1)
    ltv_asc = np.sort(ltv)
    gini = (2 * np.sum(idx * ltv_asc) / (n * np.sum(ltv_asc))) - (n + 1) / n if np.sum(ltv_asc) > 0 else 0

    st.subheader(f"{label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 1% Rev Share", f"{top_1:.1f}%")
    c2.metric("Top 5% Rev Share", f"{top_5:.1f}%")
    c3.metric("Top 10% Rev Share", f"{top_10:.1f}%")
    c4.metric("Gini Coefficient", f"{gini:.3f}")

# =====================================================================
# Summary
# =====================================================================
st.markdown("---")
st.markdown("### 💡 Diagnostic Summary")
st.markdown(
    "**If lift curves are similar across Early and Late cohorts:**  \n"
    "→ Model generalizes well. Safe to deploy for seed selection.  \n\n"
    "**If Late cohort curves are significantly lower:**  \n"
    "→ Model may be overfitting to recent training patterns. Consider:  \n"
    "- Adding more training data  \n"
    "- Retraining more frequently  \n"
    "- Using simpler features that are more stable  \n\n"
    "**If concentration risk is very high (Gini > 0.95):**  \n"
    "→ Revenue depends on a handful of whales. Model predictions for the top tier "
    "have outsized impact. Consider ensemble strategies or whale-specific models."
)
