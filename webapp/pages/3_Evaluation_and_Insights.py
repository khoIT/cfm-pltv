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
    render_sidebar, render_top_menu, render_report_md, get_data, get_active_model, convert_vnd, get_currency_info,
    format_currency, REPORTS_DIR, DATA_DIR,
    BASELINE_HEURISTICS, compute_baseline_ranking,
    list_datasets_by_role,
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

    # Precision@K / Recall@K  (high spender = top 5% by actual LTV)
    threshold = np.percentile(y_true, 95)
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

render_report_md(REPORTS_DIR / "evaluation_metrics.md", "📄 Evaluation Metrics Report")

# ── Dataset role selector ─────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Dataset")
role_meta = list_datasets_by_role()

_role_opts = [r for r in ["train", "test"] if role_meta.get(r) is not None]
if not _role_opts:
    _role_opts = ["train"]  # fallback label even if missing

_role_labels = {"train": "🏋️ Train (in-sample)", "test": "🧪 Test (holdout)"}
_eval_role = st.radio(
    "Evaluate on:",
    _role_opts,
    format_func=lambda r: _role_labels.get(r, r),
    horizontal=True,
    key="eval_role_select",
    help="**Train** = in-sample (optimistic). **Test** = held-out 20% — unbiased view of real-world performance.",
)

_role_info = role_meta.get(_eval_role)
if _role_info is not None:
    st.caption(
        f"**{_role_info['name']}** — {_role_info['size_mb']:.1f} MB"
        + (f" | {_role_info['split_info']}" if _role_info.get('split_info') else "")
    )
    df_eval = pd.read_csv(_role_info["path"], low_memory=False)
else:
    st.warning("Dataset not found — falling back to registry default.")
    df_eval = get_data()

if _eval_role == "test":
    st.info(
        "📊 **Test (holdout) mode** — these users were not seen during model training. "
        "Metrics here reflect true out-of-sample performance.",
        icon="🧪"
    )
else:
    st.info(
        "📊 **Train (in-sample) mode** — metrics may be optimistic since the model was trained on this data. "
        "Switch to **Test** for an unbiased evaluation.",
        icon="🏋️"
    )

st.caption(f"Evaluating on: **{len(df_eval):,}** rows")
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
    # Realistic demo: moderate noise so ranking isn't near-perfect
    noise_mult = rng.lognormal(0, 0.8, len(y_true))  # wide multiplicative noise
    noise_add  = rng.normal(0, np.std(y_true) * 0.3, len(y_true))  # additive noise
    y_pred_model = y_true * noise_mult + noise_add
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

# ── Oracle (perfect model) — sorted by actual ltv30 ───────────────
_oracle_order = np.argsort(-y_true)
_oracle_sorted = y_true[_oracle_order]
_total = _oracle_sorted.sum()
_oracle_cumrev = np.cumsum(_oracle_sorted) / _total if _total > 0 else np.zeros(len(_oracle_sorted))
_oracle_pcts = np.arange(1, len(_oracle_cumrev) + 1) / len(_oracle_cumrev)

# ── Random baseline cumrev ─────────────────────────────────────────
_random_pcts = np.array([0, 100])
_random_cumrev = np.array([0, 100])

scorecard_rows = []
for name, m in all_metrics.items():
    lift10_idx = min(int(len(m["lift_cumrev"]) * 0.1), len(m["lift_cumrev"]) - 1)
    oracle10_idx = min(int(len(_oracle_cumrev) * 0.1), len(_oracle_cumrev) - 1)
    scorecard_rows.append({
        "Strategy": name,
        "Spearman ρ": round(m["spearman_rho"], 3),
        "Lift@10%": f"{m['lift_cumrev'][lift10_idx]:.1%}",
        "Oracle@10%": f"{_oracle_cumrev[oracle10_idx]:.1%}",
        "Gap to Oracle": f"{(_oracle_cumrev[oracle10_idx] - m['lift_cumrev'][lift10_idx])*100:+.1f}pp",
        "Prec@5%": round(m["precision_at_k"][5], 3),
        "Recall@10%": round(m["recall_at_k"][10], 3),
        "AUC": round(m["auc"], 3) if m["auc"] is not None else "N/A",
    })
score_df = pd.DataFrame(scorecard_rows)

# ── Lift figure ────────────────────────────────────────────────────
fig_lift = go.Figure()

# Strategy lines first (so oracle renders on top)
for name, m in all_metrics.items():
    sample_idx = np.linspace(0, len(m["lift_pcts"]) - 1, 200, dtype=int)
    fig_lift.add_trace(go.Scatter(
        x=m["lift_pcts"][sample_idx] * 100, y=m["lift_cumrev"][sample_idx] * 100,
        name=name, line=dict(color=color_map.get(name, "blue"), width=2),
        hovertemplate=f"Top %{{x:.1f}}% → %{{y:.1f}}% revenue ({name})<extra></extra>",
    ))

# Oracle line on top (theoretical max) — thick dashed green
_s = np.linspace(0, len(_oracle_pcts) - 1, 200, dtype=int)
fig_lift.add_trace(go.Scatter(
    x=_oracle_pcts[_s] * 100, y=_oracle_cumrev[_s] * 100,
    name="Oracle (perfect model)",
    line=dict(color="#2ecc71", width=3, dash="dash"),
    hovertemplate="Top %{x:.1f}% → %{y:.1f}% revenue (oracle)<extra></extra>",
))

# Random diagonal
fig_lift.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], name="Random",
    line=dict(dash="dash", color="lightgray", width=1),
    hoverinfo="skip",
))

# Vertical markers at key K%
for _k in [1, 5, 10, 20]:
    fig_lift.add_vline(x=_k, line_dash="dot", line_color="rgba(150,150,150,0.4)",
                       annotation_text=f"{_k}%", annotation_position="top",
                       annotation_font_size=10)

fig_lift.update_layout(
    title="Cumulative Revenue Lift — Model vs Oracle vs Random",
    xaxis_title="% Users selected (ranked by strategy)",
    yaxis_title="% Cumulative Revenue Captured",
    xaxis=dict(ticksuffix="%", range=[0, 100]),
    yaxis=dict(ticksuffix="%", range=[0, 100]),
    height=480, legend=dict(orientation="h", y=-0.18),
    hovermode="x unified",
)

# ── Top-K comparison table ─────────────────────────────────────────
_k_breakpoints = [1, 5, 10, 20]
_topk_rows = []
for _k in _k_breakpoints:
    _idx = min(int(len(_oracle_cumrev) * _k / 100), len(_oracle_cumrev) - 1)
    row = {"Top-K %": f"Top {_k}%", "Oracle (max possible)": f"{_oracle_cumrev[_idx]:.1%}"}
    for name, m in all_metrics.items():
        _midx = min(int(len(m["lift_cumrev"]) * _k / 100), len(m["lift_cumrev"]) - 1)
        _model_val = m["lift_cumrev"][_midx]
        _gap = (_oracle_cumrev[_idx] - _model_val) * 100
        row[name] = f"{_model_val:.1%}"
        row[f"{name} gap"] = f"{_gap:+.1f}pp"
    _random_val = _k / 100
    row["Random"] = f"{_random_val:.1%}"
    _topk_rows.append(row)
_topk_df = pd.DataFrame(_topk_rows)

col_score, col_lift = st.columns([1, 1.4])
with col_score:
    st.subheader("📋 Scorecard")
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    if len(score_df) > 1:
        best_spear = score_df.loc[score_df["Spearman ρ"].idxmax(), "Strategy"]
        st.markdown(f"**Best ranking (Spearman ρ):** `{best_spear}`")
    st.markdown("---")
    st.markdown("**📊 Cumulative Revenue @ Key Thresholds**")
    st.caption("How much revenue is captured by selecting the top K% of users, ranked by each strategy vs the theoretical maximum (Oracle = sorted by actual LTV30).")
    st.dataframe(_topk_df, use_container_width=True, hide_index=True)
with col_lift:
    st.subheader("Lift Curve — Model vs Oracle vs Random")
    st.plotly_chart(fig_lift, use_container_width=True)

# ── Precision@K & Recall@K ─────────────────────────────────────────
st.subheader("Precision@K & Recall@K")
st.markdown("> **Precision@K**: Of the top K% users you pick, how many are truly high-value (top 5% by actual LTV30)?  \n"
            "> **Recall@K**: Of all high-value users (top 5%), what fraction did you capture in your top K%?")

col1, col2 = st.columns(2)

# Build grouped bar data — use string labels so Plotly spaces bars evenly
k_vals = [1, 5, 10, 20]
prec_rows, rec_rows = [], []
for name, m in all_metrics.items():
    for k in k_vals:
        prec_rows.append({"K (%)": f"Top {k}%", "Precision": round(m["precision_at_k"][k], 4), "Strategy": name})
        rec_rows.append({"K (%)": f"Top {k}%", "Recall": round(m["recall_at_k"][k], 4), "Strategy": name})

# Preserve order
k_order = [f"Top {k}%" for k in k_vals]

with col1:
    fig_pk = px.bar(
        pd.DataFrame(prec_rows), x="K (%)", y="Precision", color="Strategy",
        barmode="group", title="Precision@K (high-value = top 5%)",
        color_discrete_map=color_map,
        category_orders={"K (%)": k_order},
        text_auto=".2f",
    )
    _pk_max = pd.DataFrame(prec_rows)["Precision"].max()
    fig_pk.update_layout(
        yaxis_tickformat=".0%", yaxis_title="Precision", height=420,
        yaxis_range=[0, min(_pk_max * 1.25, 1.15)],
        margin=dict(t=60),
    )
    fig_pk.update_traces(textposition="outside", textfont_size=11)
    st.plotly_chart(fig_pk, use_container_width=True)
with col2:
    fig_rk = px.bar(
        pd.DataFrame(rec_rows), x="K (%)", y="Recall", color="Strategy",
        barmode="group", title="Recall@K — of all high-value users, how many did you capture?",
        color_discrete_map=color_map,
        category_orders={"K (%)": k_order},
        text_auto=".2f",
    )
    _rk_max = pd.DataFrame(rec_rows)["Recall"].max()
    fig_rk.update_layout(
        yaxis_tickformat=".0%", yaxis_title="Recall", height=420,
        yaxis_range=[0, min(_rk_max * 1.25, 1.15)],
        margin=dict(t=60),
    )
    fig_rk.update_traces(textposition="outside", textfont_size=11)
    st.plotly_chart(fig_rk, use_container_width=True)

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
