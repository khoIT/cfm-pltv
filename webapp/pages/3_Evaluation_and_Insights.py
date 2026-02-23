"""
Page 3 — Evaluation & Insights
Lift curve, Precision/Recall@K, Spearman, Calibration, ROC/AUC.
Compare XGBoost model vs simple baseline heuristics.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, render_top_menu, get_data, get_active_model, convert_vnd, get_currency_info,
    format_currency, REPORTS_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking,
)

render_top_menu()
render_sidebar()

# ── helpers ────────────────────────────────────────────────────────
def compute_eval_metrics(y_true, y_pred, label="Model"):
    """Compute all evaluation metrics from predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    results = {"label": label}

    # Spearman
    rho, _ = spearmanr(y_true, y_pred)
    results["spearman_rho"] = rho

    # Lift curve
    order = np.argsort(-y_pred)
    sorted_actual = y_true[order]
    total = sorted_actual.sum()
    cumrev = np.cumsum(sorted_actual) / total if total > 0 else np.zeros(len(sorted_actual))
    pcts = np.arange(1, len(cumrev) + 1) / len(cumrev)
    results["lift_pcts"] = pcts
    results["lift_cumrev"] = cumrev

    # Precision@K / Recall@K  (high spender = top 10% by actual LTV)
    threshold = np.percentile(y_true, 90)
    is_high = (y_true >= threshold).astype(int)
    total_high = is_high.sum()

    prec_at_k, recall_at_k = {}, {}
    for k_pct in [1, 5, 10, 20]:
        k = max(1, int(len(y_pred) * k_pct / 100))
        top_k_idx = order[:k]
        hits = is_high[top_k_idx].sum()
        prec_at_k[k_pct] = hits / k if k > 0 else 0
        recall_at_k[k_pct] = hits / total_high if total_high > 0 else 0
    results["precision_at_k"] = prec_at_k
    results["recall_at_k"] = recall_at_k

    # Calibration
    n_bins = 10
    pred_range = y_pred.max() - y_pred.min()
    if pred_range > 0:
        bin_idx = np.digitize(y_pred, np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    else:
        bin_idx = np.zeros(len(y_pred), dtype=int)
    cal_pred, cal_actual = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            cal_pred.append(float(y_pred[mask].mean()))
            cal_actual.append(float(y_true[mask].mean()))
    results["cal_pred"] = cal_pred
    results["cal_actual"] = cal_actual

    # ROC/AUC
    is_payer = (y_true > 0).astype(int)
    if len(np.unique(is_payer)) > 1:
        fpr, tpr, _ = roc_curve(is_payer, y_pred)
        results["fpr"] = fpr
        results["tpr"] = tpr
        results["auc"] = auc(fpr, tpr)
    else:
        results["auc"] = None

    return results


# ── page ─────────────────────────────────────────────────────────────────
st.title("📊 Evaluation & Insights")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No dataset selected")
    st.info("Please select a dataset from the **Dataset Registry** in the sidebar.")
    st.stop()

cur = get_currency_info()

report_path = REPORTS_DIR / "evaluation_metrics.md"
if report_path.exists():
    with st.expander("📄 Evaluation Metrics Report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))

# Load dataset from registry
df_eval = get_data()
st.caption(f"Dataset: **{len(df_eval):,}** rows (from Dataset Registry)")
y_true = df_eval["ltv30"].values

# ── model predictions ─────────────────────────────────────────────
model, model_feats = get_active_model()
use_live = model is not None

if use_live:
    from sklearn.preprocessing import LabelEncoder
    num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
    cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]

    # Prepare test features using same encoding as training
    feature_df_test = df_eval[num_feats + cat_feats].copy()
    for c in cat_feats:
        le = LabelEncoder()
        feature_df_test[c] = le.fit_transform(feature_df_test[c].astype(str))
    if "first_charge_day_offset_d7" in feature_df_test.columns:
        feature_df_test["first_charge_day_offset_d7"] = feature_df_test["first_charge_day_offset_d7"].fillna(-1)

    y_pred_log = model.predict(feature_df_test)
    y_pred_model = np.expm1(y_pred_log)
    model_label = f"XGBoost ({len(model_feats)}f)"
else:
    st.info("⬅️ No trained model found. Train one on the **Features & Model** page.  \n"
            "Meanwhile, you can still compare **baseline heuristics** below.")
    rng = np.random.default_rng(42)
    y_pred_model = y_true * rng.uniform(0.6, 1.4, len(y_true)) + rng.normal(0, 0.5, len(y_true))
    y_pred_model = np.maximum(y_pred_model, 0)
    model_label = "XGBoost (demo)"

# =====================================================================
# BASELINE SELECTOR — prominent toggle panel
# =====================================================================
st.header("🔀 Compare: Model vs Baseline Heuristics")
st.markdown(
    "Toggle **ON** the heuristics below to overlay them on every chart.  \n"
    "Each baseline simply **ranks users by a single column** — no ML needed.  \n"
    "This shows whether the model adds value over a naive rule."
)

# Render toggles in a prominent row
cols = st.columns(len(BASELINE_HEURISTICS))
active_baselines = {}
for i, (name, info) in enumerate(BASELINE_HEURISTICS.items()):
    with cols[i]:
        on = st.toggle(name.split(" (")[0], value=(name == "rev_d7 (D7 Revenue)"), key=f"bl_{name}",
                        help=info["description"])
        if on:
            bl_scores = compute_baseline_ranking(df_eval, info["column"])
            active_baselines[name] = {
                "scores": bl_scores[:len(y_true)],
                "color": info["color"],
            }

# ── compute metrics for all strategies ─────────────────────────────
all_metrics = {}
all_metrics[model_label] = compute_eval_metrics(y_true, y_pred_model, label=model_label)
for bl_name, bl_info in active_baselines.items():
    all_metrics[bl_name] = compute_eval_metrics(y_true, bl_info["scores"], label=bl_name)

# ── Colors ─────────────────────────────────────────────────────────
strategy_colors = {"royalblue": model_label}
color_map = {model_label: "royalblue"}
for bl_name, bl_info in active_baselines.items():
    color_map[bl_name] = bl_info["color"]

# ── Scorecard + Lift Curve (side-by-side) ─────────────────────────
st.markdown("---")

scorecard_rows = []
for name, m in all_metrics.items():
    lift10_idx = int(len(m["lift_cumrev"]) * 0.1)
    scorecard_rows.append({
        "Strategy": name,
        "Spearman ρ": round(m["spearman_rho"], 3),
        "Lift@10%": f"{m['lift_cumrev'][lift10_idx]:.1%}",
        "Prec@5%": round(m["precision_at_k"][5], 3),
        "Recall@10%": round(m["recall_at_k"][10], 3),
        "AUC": round(m["auc"], 3) if m["auc"] is not None else "N/A",
    })
score_df = pd.DataFrame(scorecard_rows)

fig_lift = go.Figure()
for name, m in all_metrics.items():
    sample_idx = np.linspace(0, len(m["lift_pcts"]) - 1, 200, dtype=int)
    fig_lift.add_trace(go.Scatter(
        x=m["lift_pcts"][sample_idx] * 100, y=m["lift_cumrev"][sample_idx] * 100,
        name=name, line=dict(color=color_map.get(name, "blue"), width=2),
    ))
fig_lift.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], name="Random", line=dict(dash="dash", color="lightgray"),
))
fig_lift.update_layout(
    xaxis_title="% Users (ranked by strategy)",
    yaxis_title="% Cumulative Revenue Captured",
    height=450, legend=dict(orientation="h", y=-0.15),
)

col_score, col_lift = st.columns([1, 1.3])
with col_score:
    st.subheader("📋 Scorecard")
    st.dataframe(score_df, use_container_width=True, hide_index=True, height=420)
    if len(score_df) > 1:
        best_spear = score_df.loc[score_df["Spearman ρ"].idxmax(), "Strategy"]
        st.markdown(f"**Best ranking (Spearman ρ):** `{best_spear}`")
with col_lift:
    st.subheader("Lift Curve")
    st.plotly_chart(fig_lift, use_container_width=True)

# ── Precision@K & Recall@K ─────────────────────────────────────────
st.subheader("Precision@K & Recall@K")
st.markdown("> **Precision@K**: Of the top K% users you pick, how many are truly high-value (top 10%)?  \n"
            "> **Recall@K**: Of all high-value users, what fraction did you capture in your top K%?")

col1, col2 = st.columns(2)

# Build grouped bar data
k_vals = [1, 5, 10, 20]
prec_rows, rec_rows = [], []
for name, m in all_metrics.items():
    for k in k_vals:
        prec_rows.append({"K (%)": k, "Precision": m["precision_at_k"][k], "Strategy": name})
        rec_rows.append({"K (%)": k, "Recall": m["recall_at_k"][k], "Strategy": name})

with col1:
    fig_pk = px.bar(pd.DataFrame(prec_rows), x="K (%)", y="Precision", color="Strategy",
                    barmode="group", title="Precision@K (high spender = top 10%)",
                    color_discrete_map=color_map)
    st.plotly_chart(fig_pk, width='stretch')
with col2:
    fig_rk = px.bar(pd.DataFrame(rec_rows), x="K (%)", y="Recall", color="Strategy",
                    barmode="group", title="Recall@K",
                    color_discrete_map=color_map)
    st.plotly_chart(fig_rk, width='stretch')

# ── Calibration + ROC (side-by-side) ──────────────────────────────
m0 = all_metrics[model_label]
any_auc = any(m["auc"] is not None for m in all_metrics.values())

if (m0["cal_pred"] and m0["cal_actual"]) or any_auc:
    col_cal, col_roc = st.columns(2)
    
    with col_cal:
        st.subheader("Calibration Plot")
        if m0["cal_pred"] and m0["cal_actual"]:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=m0["cal_pred"], y=m0["cal_actual"],
                mode="markers+lines", name=model_label, marker=dict(size=8, color="royalblue"),
            ))
            max_val = max(max(m0["cal_pred"]), max(m0["cal_actual"]))
            fig_cal.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], name="Perfect", line=dict(dash="dash", color="gray"),
            ))
            fig_cal.update_layout(
                xaxis_title=f"Predicted LTV30 ({cur['symbol']})",
                yaxis_title=f"Actual LTV30 ({cur['symbol']})",
                height=420,
            )
            st.plotly_chart(fig_cal, use_container_width=True)
    
    with col_roc:
        st.subheader("ROC Curve")
        if any_auc:
            fig_roc = go.Figure()
            for name, m in all_metrics.items():
                if m["auc"] is not None:
                    fig_roc.add_trace(go.Scatter(
                        x=m["fpr"], y=m["tpr"],
                        name=f"{name} (AUC={m['auc']:.3f})",
                        line=dict(color=color_map.get(name, "blue"), width=2),
                    ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="lightgray"),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=420, legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ── Educational takeaway ───────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Key Takeaway")
st.markdown(
    "If the **XGBoost model** beats every single-feature baseline on Lift, Precision, and Spearman, "
    "it means the model is learning **non-obvious signal combinations** that no single heuristic captures.  \n\n"
    "If a baseline like `rev_d7` is almost as good, then a simple rule might be sufficient — "
    "saving you the complexity of maintaining a full ML pipeline."
)
