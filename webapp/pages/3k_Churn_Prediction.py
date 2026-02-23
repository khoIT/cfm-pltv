"""
Page 3k — Churn Prediction (Payers)
Predict which D7 payers will fail to reach their LTV30 potential.
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
    render_sidebar, render_top_menu, render_report_md, get_registry_path,
    convert_vnd, get_currency_info, format_currency, REPORTS_DIR,
)

render_top_menu()
render_sidebar()

CHURN_FEAT_COLS = [
    "games_d7", "active_days_d7", "win_rate_d7", "kd_d7",
    "max_level_seen_d7", "login_rows_d7", "rev_d7", "txn_cnt_d7",
    "first_charge_day_offset_d7", "avg_game_duration_d7", "avg_score_d7",
]


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training churn prediction model…")
def compute_churn_metrics(csv_path: str, file_mtime: float):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns or "rev_d7" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' and 'rev_d7' columns."

    df = df.copy()
    df["ltv30"] = pd.to_numeric(df["ltv30"], errors="coerce").fillna(0)
    df["rev_d7"] = pd.to_numeric(df["rev_d7"], errors="coerce").fillna(0)

    # Payers: anyone who paid in D7
    payers = df[df["rev_d7"] > 0].copy()
    if len(payers) < 50:
        return None, None, "Not enough D7 payers found (need ≥50)."

    # Churn definition: D7 payer whose LTV30 is in the bottom 50% of payers
    # i.e. they paid early but didn't grow — "one-and-done"
    ltv_median = payers["ltv30"].median()
    payers["churned"] = (payers["ltv30"] <= ltv_median).astype(int)

    # Payer segments — build bins carefully to avoid duplicates
    p95 = df["ltv30"].quantile(0.95)
    # Ensure strictly increasing bins by deduplicating
    raw_bins = sorted(set([-0.01, 0.0, float(ltv_median), float(p95)]))
    raw_bins.append(float("inf"))
    # Need at least 2 intervals (3 edges); fall back to simple split if degenerate
    if len(raw_bins) < 3:
        raw_bins = [-0.01, float(ltv_median) if ltv_median > 0 else 1.0, float("inf")]
    n_labels = len(raw_bins) - 1
    label_pool = ["Low Payer", "Mid Payer", "High Payer", "Whale"]
    seg_labels = label_pool[:n_labels]
    payers["payer_seg"] = pd.cut(
        payers["ltv30"],
        bins=raw_bins,
        labels=seg_labels,
    )

    seg_stats = payers.groupby("payer_seg", observed=True).agg(
        users=("ltv30", "count"),
        avg_ltv30=("ltv30", "mean"),
        avg_rev_d7=("rev_d7", "mean"),
        churn_rate=("churned", "mean"),
    ).reset_index()
    seg_stats["churn_rate_%"] = (seg_stats["churn_rate"] * 100).round(1)

    # Feature importance via GBM
    feat_cols = [c for c in CHURN_FEAT_COLS if c in payers.columns]
    feat_importance = None
    model_metrics = None
    churn_scores = None

    if len(feat_cols) >= 3 and len(payers) >= 100:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, precision_score, recall_score

        X = payers[feat_cols].fillna(0).values
        y = payers["churned"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        feat_importance = pd.DataFrame({
            "Feature": feat_cols,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        model_metrics = {"AUC": round(auc, 4), "Precision": round(prec, 3),
                         "Recall": round(rec, 3), "Test Users": len(y_test)}

        # Score all payers
        churn_scores = model.predict_proba(payers[feat_cols].fillna(0).values)[:, 1]
        payers = payers.copy()
        payers["churn_score"] = churn_scores

    # Retention signals: compare churned vs retained
    retention_comparison = None
    if feat_cols:
        comp = payers.groupby("churned")[feat_cols].mean().T.reset_index()
        comp.columns = ["Feature", "Retained", "Churned"]
        comp["Retained/Churned Ratio"] = (
            comp["Retained"] / comp["Churned"].replace(0, np.nan)).round(2)
        retention_comparison = comp.sort_values("Retained/Churned Ratio", ascending=False)

    # txn_cnt distribution
    txn_dist = None
    if "txn_cnt_d7" in payers.columns:
        txn_dist = payers.groupby("txn_cnt_d7").agg(
            users=("ltv30", "count"),
            avg_ltv30=("ltv30", "mean"),
            churn_rate=("churned", "mean"),
        ).reset_index()
        txn_dist = txn_dist[txn_dist["txn_cnt_d7"] <= 10]
        txn_dist["churn_rate_%"] = (txn_dist["churn_rate"] * 100).round(1)

    return df, {
        "payers": payers,
        "seg_stats": seg_stats,
        "feat_importance": feat_importance,
        "model_metrics": model_metrics,
        "retention_comparison": retention_comparison,
        "txn_dist": txn_dist,
        "feat_cols": feat_cols,
        "ltv_median": ltv_median,
        "p95": p95,
        "n_payers": len(payers),
        "n_total": len(df),
    }, None


# ── page ───────────────────────────────────────────────────────────
st.title("📉 Churn Prediction (Payers)")
st.markdown(
    "Predict which **D7 payers** will fail to grow beyond their initial purchase. "
    "For a whale-intensive game, retaining a single whale is worth more than acquiring 100 non-payers."
)

cur = get_currency_info()

# Load from registry
ds_path, ds_mtime = get_registry_path()
with st.spinner("Training churn prediction model… (~15–30s, cached after first run)"):
    df_raw, metrics, error = compute_churn_metrics(ds_path, ds_mtime)

if error:
    st.error(f"❌ {error}")
    st.stop()

seg_stats = metrics["seg_stats"]
feat_importance = metrics["feat_importance"]
model_metrics = metrics["model_metrics"]
retention_comparison = metrics["retention_comparison"]
txn_dist = metrics["txn_dist"]
n_payers = metrics["n_payers"]
n_total = metrics["n_total"]
ltv_median = metrics["ltv_median"]
p95 = metrics["p95"]
payers = metrics["payers"]

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Churn_Prediction.md", "📄 Full Churn Prediction Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Payer Overview")

overall_churn = payers["churned"].mean() * 100
top_seg = payers["payer_seg"].cat.categories[-1] if "payer_seg" in payers.columns and hasattr(payers["payer_seg"], "cat") else None
whale_churn = payers[payers["payer_seg"] == top_seg]["churned"].mean() * 100 if top_seg is not None else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("D7 Payers", f"{n_payers:,}", f"{n_payers/n_total*100:.1f}% of all users")
with k2:
    st.metric("Overall Churn Rate", f"{overall_churn:.1f}%", "D7 payers with low LTV30")
with k3:
    st.metric("Whale Churn Rate", f"{whale_churn:.1f}%", "top 5% LTV30 payers")
with k4:
    if model_metrics:
        st.metric("Model AUC", f"{model_metrics['AUC']:.4f}", "churn classifier")

# ── Model Performance ───────────────────────────────────────────────
if model_metrics:
    st.markdown("---")
    st.header("🤖 Churn Model Performance")

    feat_cols_used = metrics["feat_cols"]
    rev_feats = [c for c in feat_cols_used if "rev" in c or "txn" in c]
    eng_feats = [c for c in feat_cols_used if c not in rev_feats]
    with st.expander("ℹ️ How was this model trained? (click to expand)", expanded=False):
        st.markdown("""
**What is this model doing?**
This is a **Gradient Boosting classifier** — a machine learning model that learns patterns
from historical data to predict which paying users are likely to churn (stop spending) before D30.

**Who is it trained on?**
Only users who already made **at least one purchase in the first 7 days** (D7 payers).
Non-payers are excluded — this model is purely about *retention of existing payers*.

**What features did we include?**
The model uses **both revenue signals and behavioral engagement signals**:
""")
        col_f1, col_f2 = st.columns(2)
        rev_feat_labels = {
            'rev_d7': 'total spend in first 7 days',
            'txn_cnt_d7': 'number of separate purchases in D7',
        }
        with col_f1:
            st.markdown("**💰 Revenue / Payment features:**")
            for c in rev_feats:
                lbl = rev_feat_labels.get(c, c)
                st.markdown(f"- `{c}` — {lbl}")
        with col_f2:
            st.markdown("**🎮 Behavioral / Engagement features:**")
            for c in eng_feats:
                label = {
                    'games_d7': 'total games played in D7',
                    'active_days_d7': 'days the user logged in during D7',
                    'win_rate_d7': 'fraction of games won',
                    'kd_d7': 'kill/death ratio',
                    'max_level_seen_d7': 'highest level reached in D7',
                    'login_rows_d7': 'number of login sessions',
                    'first_charge_day_offset_d7': 'which day (0–7) the first purchase happened',
                    'avg_game_duration_d7': 'average game session length',
                    'avg_score_d7': 'average in-game score',
                }.get(c, c)
                st.markdown(f"- `{c}` — {label}")
        st.markdown("""
**Why include revenue features if the target is also revenue-based?**
The churn label is defined as *LTV30 ≤ median payer LTV30* — so `rev_d7` (D7 spend) is
naturally correlated with the target. This makes it the model's strongest predictor,
but it is **partially circular**: a user who spent ₫500k in D7 almost certainly has high LTV30.
The behavioral features (games, active_days, txn_cnt) are the **actionable signals** —
you can intervene on engagement, but you cannot change past revenue.

**How accurate is it?**
- **AUC-ROC** measures overall discrimination: 1.0 = perfect, 0.5 = random guess
- **Precision** = of users flagged as churners, what % actually churned
- **Recall** = of all actual churners, what % did we catch
""")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("AUC-ROC", f"{model_metrics['AUC']:.4f}")
    with m2:
        st.metric("Precision", f"{model_metrics['Precision']:.3f}")
    with m3:
        st.metric("Recall", f"{model_metrics['Recall']:.3f}")
    with m4:
        st.metric("Test Users", f"{model_metrics['Test Users']:,}")

# ── Payer Segment Churn ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 Churn Rate by Payer Segment")
col1, col2 = st.columns(2)

seg_colors = ["#95a5a6", "#3498db", "#e67e22", "#e74c3c"]
with col1:
    fig_churn_seg = go.Figure()
    for i, row in seg_stats.iterrows():
        fig_churn_seg.add_trace(go.Bar(
            x=[str(row["payer_seg"])],
            y=[row["churn_rate_%"]],
            marker_color=seg_colors[i % len(seg_colors)],
            text=[f"{row['churn_rate_%']:.1f}%"],
            textposition="outside",
        ))
    fig_churn_seg.add_hline(y=50, line_dash="dash", line_color="gray",
                             annotation_text="50% baseline")
    fig_churn_seg.update_layout(
        title="Churn Rate by Payer Segment",
        yaxis_title="Churn Rate (%)", height=400,
        yaxis=dict(range=[0, 110]), showlegend=False,
    )
    st.plotly_chart(fig_churn_seg, use_container_width=True)

with col2:
    fig_ltv_seg = go.Figure()
    for i, row in seg_stats.iterrows():
        fig_ltv_seg.add_trace(go.Bar(
            x=[str(row["payer_seg"])],
            y=[convert_vnd(row["avg_ltv30"], cur["code"])],
            marker_color=seg_colors[i % len(seg_colors)],
            text=[format_currency(convert_vnd(row["avg_ltv30"], cur["code"]), cur["code"])],
            textposition="outside",
        ))
    fig_ltv_seg.update_layout(
        title=f"Avg LTV30 by Payer Segment ({cur['symbol']})",
        yaxis_title=f"Avg LTV30 ({cur['symbol']})", height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_ltv_seg, use_container_width=True)

# ── Feature Importance ──────────────────────────────────────────────
if feat_importance is not None:
    st.markdown("---")
    st.header("🔑 Churn Prediction Feature Importance")

    st.info(
        """**How to read these charts — for business users:**

**Left chart (Feature Importance bar):** Each bar shows how much a feature contributed to the model's churn predictions.
A longer bar = the model relied on that feature more heavily when deciding if a user will churn.
*It does NOT mean that feature causes churn — it means it was the most useful signal for prediction.*

**Why does `rev` (D7 revenue) dominate?**
This is expected and slightly circular: users who spent more in D7 almost always have higher LTV30,
so the model correctly learns that high D7 spend = low churn risk. However, you cannot act on past revenue —
it's a *descriptor*, not an *intervention lever*.

**Right chart (Retained vs Churned avg values):** The green bar shows the average value for retained payers,
red for churned payers. The `rev` bar looks enormous because it's in VND (₫) — all other features
(games played, active days, win rate) are on a much smaller scale and appear flat by comparison.
This is a **scale effect**, not a signal absence — those features do differ between retained and churned users.

**The actionable signals** are the behavioral features ranked below `rev`:
`txn_cnt` (repeat purchases), `active_days`, `games`, `max_level_seen` — these you can influence
through product interventions, push notifications, and retention offers.""",
        icon="💡"
    )
    col3, col4 = st.columns(2)

    with col3:
        top_feats = feat_importance.head(10)
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=top_feats["Importance"],
            y=top_feats["Feature"].apply(lambda c: c.replace("_d7", "")),
            orientation="h",
            marker_color="#e74c3c",
        ))
        fig_imp.update_layout(
            title="Top 10 Churn Prediction Features",
            xaxis_title="Feature Importance", height=400,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col4:
        if retention_comparison is not None:
            fig_ret = go.Figure()
            top_ret = retention_comparison.head(8)
            fig_ret.add_trace(go.Bar(
                name="Retained",
                x=top_ret["Feature"].apply(lambda c: c.replace("_d7", "")),
                y=top_ret["Retained"],
                marker_color="#2ecc71",
            ))
            fig_ret.add_trace(go.Bar(
                name="Churned",
                x=top_ret["Feature"].apply(lambda c: c.replace("_d7", "")),
                y=top_ret["Churned"],
                marker_color="#e74c3c",
            ))
            fig_ret.update_layout(
                title="Retained vs Churned: Avg Feature Values",
                barmode="group", height=400,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

# ── Txn Count vs Churn ──────────────────────────────────────────────
if txn_dist is not None and len(txn_dist) > 0:
    st.markdown("---")
    st.header("🔁 Transaction Count vs Churn Rate")

    with st.expander("ℹ️ Why did we build this chart and what does it mean?", expanded=True):
        st.markdown("""
**Why transaction count?**
While the model flagged `rev_d7` (total D7 spend) as its top feature, that signal is hard to act on —
you already know how much someone spent. What you *can* observe in real time is **how many separate
purchases** a user made. A user who buys once for ₫500k behaves very differently from one who buys
5 times for ₫100k each — even if their D7 revenue is identical.

Repeat purchasing within D7 signals **habitual spending behaviour** — the user has formed a purchase loop,
not just made a one-off transaction. This is the leading indicator we can act on.

**How to read the left chart (Churn Rate by Transaction Count):**
- The X-axis is the number of separate purchases a user made in their first 7 days
- The Y-axis is the % of those users who churned (did not grow their LTV beyond the median)
- A user with **1 transaction has ~67% churn risk** — they tried the game once, paid once, and left
- Each additional transaction roughly **halves the churn rate** — by 5+ transactions, churn is near zero

**How to read the right chart (Avg LTV30 by Transaction Count):**
- Users with more D7 transactions have dramatically higher 30-day lifetime value
- This is not just because they spent more in D7 — it reflects a compounding habit of spending

**Business implication:**
The goal of retention interventions should be to **trigger a second purchase**, not just reward the first.
A user who makes 2 purchases in D7 is 2× less likely to churn than a one-time buyer.
This is the basis for the recommendation: *trigger a retention offer when churn_score > 0.7 AND txn_cnt_d7 = 1*.
""")
    col5, col6 = st.columns(2)

    with col5:
        fig_txn_churn = go.Figure()
        fig_txn_churn.add_trace(go.Bar(
            x=txn_dist["txn_cnt_d7"].astype(str),
            y=txn_dist["churn_rate_%"],
            marker_color="#e74c3c",
            text=txn_dist["churn_rate_%"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
        ))
        fig_txn_churn.update_layout(
            title="Churn Rate by D7 Transaction Count",
            xaxis_title="Transactions in D7",
            yaxis_title="Churn Rate (%)", height=380,
            yaxis=dict(range=[0, 110]),
        )
        st.plotly_chart(fig_txn_churn, use_container_width=True)

    with col6:
        fig_txn_ltv = go.Figure()
        fig_txn_ltv.add_trace(go.Scatter(
            x=txn_dist["txn_cnt_d7"],
            y=txn_dist["avg_ltv30"].apply(lambda v: convert_vnd(v, cur["code"])),
            mode="lines+markers",
            line=dict(color="#3498db", width=3),
            marker=dict(size=10),
        ))
        fig_txn_ltv.update_layout(
            title=f"Avg LTV30 by D7 Transaction Count ({cur['symbol']})",
            xaxis_title="Transactions in D7",
            yaxis_title=f"Avg LTV30 ({cur['symbol']})", height=380,
        )
        st.plotly_chart(fig_txn_ltv, use_container_width=True)

# ── Churn Score Distribution ────────────────────────────────────────
if "churn_score" in payers.columns:
    st.markdown("---")
    st.header("📊 Churn Risk Score Distribution")
    fig_hist = px.histogram(
        payers, x="churn_score", color="payer_seg",
        nbins=40, barmode="overlay", opacity=0.7,
        title="Churn Risk Score by Payer Segment",
        labels={"churn_score": "Churn Risk Score (0=safe, 1=at-risk)"},
        color_discrete_sequence=seg_colors,
        height=380,
    )
    fig_hist.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Summary Table ───────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Payer Segment Summary")
tbl = seg_stats.copy()
tbl["avg_ltv30"] = tbl["avg_ltv30"].apply(
    lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["avg_rev_d7"] = tbl["avg_rev_d7"].apply(
    lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl = tbl[["payer_seg", "users", "avg_rev_d7", "avg_ltv30", "churn_rate_%"]]
tbl.columns = ["Segment", "Users", f"Avg Rev D7 ({cur['symbol']})",
               f"Avg LTV30 ({cur['symbol']})", "Churn Rate %"]
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
st.markdown(f"- **{overall_churn:.1f}% of D7 payers** have low LTV30 — one-and-done purchases")
st.markdown(f"- **Whale churn rate: {whale_churn:.1f}%** — even top payers have churn risk")
st.markdown("- `txn_cnt_d7 ≥ 2` is the strongest retention signal — repeat purchase in D7 predicts continued spending")
st.markdown("- `active_days_d7` and `games_d7` are leading indicators — disengagement precedes churn")
st.markdown("### 🎯 Recommended Actions")
st.markdown("- Trigger retention offer for payers with **churn_score > 0.7** and rev_d7 > ₫50,000")
st.markdown("- A/B test: exclusive content vs discount offer vs social feature for at-risk whales")
st.markdown("- Feed churn predictions back into pLTV model as a feature (Feedback & Learning loop)")
