"""
Page 3h — Whale Segmentation
Profile revenue tiers and identify early whale signals using D7 behavioral data.
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

WHALE_FEAT_COLS = [
    "games_d7", "active_days_d7", "win_rate_d7", "kd_d7", "avg_score_d7",
    "max_level_seen_d7", "login_rows_d7", "rev_d7", "txn_cnt_d7",
    "first_charge_day_offset_d7",
]


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Segmenting whale tiers…")
def compute_whale_metrics(csv_path: str, file_mtime: float):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' column."

    df = df.copy()
    df["ltv30"] = pd.to_numeric(df["ltv30"], errors="coerce").fillna(0)

    total_rev = df["ltv30"].sum()
    n = len(df)

    # Define whale tiers by LTV30 percentile
    p99 = df["ltv30"].quantile(0.99)
    p95 = df["ltv30"].quantile(0.95)
    p80 = df["ltv30"].quantile(0.80)

    def tier(v):
        if v >= p99:   return "Mega-Whale (top 1%)"
        if v >= p95:   return "Whale (1–5%)"
        if v >= p80:   return "Minnow (5–20%)"
        if v > 0:      return "Low Payer (20–100%)"
        return "Non-Payer"

    tier_order = ["Mega-Whale (top 1%)", "Whale (1–5%)", "Minnow (5–20%)",
                  "Low Payer (20–100%)", "Non-Payer"]
    df["whale_tier"] = df["ltv30"].apply(tier)
    df["whale_tier"] = pd.Categorical(df["whale_tier"], categories=tier_order, ordered=True)

    # Revenue concentration
    conc = []
    for pct in [1, 5, 10, 20, 50]:
        top_n = max(1, int(n * pct / 100))
        rev_share = df["ltv30"].nlargest(top_n).sum() / total_rev * 100 if total_rev > 0 else 0
        conc.append({"Top %": f"Top {pct}%", "Users": top_n, "Revenue Share %": round(rev_share, 1)})
    conc_df = pd.DataFrame(conc)

    # Tier summary
    feat_cols = [c for c in WHALE_FEAT_COLS if c in df.columns]
    agg_dict = {"ltv30": ["count", "mean", "sum"]}
    for c in feat_cols:
        agg_dict[c] = "mean"
    tier_stats = df.groupby("whale_tier", observed=True).agg(agg_dict)
    tier_stats.columns = ["_".join(c).strip("_") for c in tier_stats.columns]
    tier_stats = tier_stats.reset_index()
    tier_stats["rev_share_%"] = (tier_stats["ltv30_sum"] / total_rev * 100).round(1)

    # K-Means clustering on engagement features (exclude payment for pure behavioral clustering)
    cluster_features = [c for c in ["games_d7", "active_days_d7", "win_rate_d7",
                                     "kd_d7", "max_level_seen_d7", "login_rows_d7"] if c in df.columns]
    cluster_df = None
    pca_df = None
    if len(cluster_features) >= 3:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        X = df[cluster_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        df["cluster"] = km.fit_predict(X_scaled)

        # Per-cluster stats with all engagement features
        agg_cl = {
            "users": ("ltv30", "count"),
            "avg_ltv30": ("ltv30", "mean"),
            "payer_rate": ("ltv30", lambda x: (x > 0).mean()),
            "whale_rate": ("ltv30", lambda x: (x >= p95).mean()),
            "total_ltv30": ("ltv30", "sum"),
        }
        for c in cluster_features:
            agg_cl[f"avg_{c}"] = (c, "mean")
        cluster_df = df.groupby("cluster").agg(**agg_cl).reset_index()
        cluster_df = cluster_df.sort_values("avg_ltv30", ascending=False).reset_index(drop=True)

        # Auto-label clusters based on dominant traits
        labels = []
        for _, row in cluster_df.iterrows():
            traits = []
            if "avg_games_d7" in row and row["avg_games_d7"] > cluster_df["avg_games_d7"].median() * 1.3:
                traits.append("High-Activity")
            elif "avg_games_d7" in row and row["avg_games_d7"] < cluster_df["avg_games_d7"].median() * 0.5:
                traits.append("Low-Activity")
            if "avg_win_rate_d7" in row and row["avg_win_rate_d7"] > cluster_df["avg_win_rate_d7"].median() * 1.2:
                traits.append("Skilled")
            if "avg_active_days_d7" in row and row["avg_active_days_d7"] > cluster_df["avg_active_days_d7"].median() * 1.3:
                traits.append("Engaged")
            if row["whale_rate"] > cluster_df["whale_rate"].median() * 2:
                traits.append("Whale-Rich")
            if not traits:
                traits.append("Average")
            labels.append(" / ".join(traits[:2]))
        cluster_df["cluster_label"] = labels
        cluster_df["rev_share_%"] = (cluster_df["total_ltv30"] / total_rev * 100).round(1)

        # PCA 2D projection for visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        # Sample for performance (max 5000 points)
        sample_n = min(5000, len(df))
        sample_idx = np.random.RandomState(42).choice(len(df), sample_n, replace=False)
        pca_df = pd.DataFrame({
            "PC1": coords[sample_idx, 0],
            "PC2": coords[sample_idx, 1],
            "cluster": df["cluster"].values[sample_idx],
            "whale_tier": df["whale_tier"].values[sample_idx],
            "ltv30": df["ltv30"].values[sample_idx],
        })
        # Map cluster IDs to labels
        cl_map = dict(zip(cluster_df["cluster"], cluster_df["cluster_label"]))
        pca_df["cluster_label"] = pca_df["cluster"].map(cl_map)
        pca_explained = pca.explained_variance_ratio_

    # D1 early detection: can we predict whale tier from limited signals?
    early_signals = [c for c in ["games_d7", "active_days_d7", "first_charge_day_offset_d7",
                                  "rev_d7", "login_rows_d7"] if c in df.columns]
    early_df = None
    if len(early_signals) >= 2:
        df["is_whale"] = (df["ltv30"] >= p95).astype(int)
        early_df = df.groupby("is_whale")[early_signals].mean().T.reset_index()
        early_df.columns = ["Feature", "Non-Whale", "Whale"]
        early_df["Whale/Non-Whale Ratio"] = (early_df["Whale"] / early_df["Non-Whale"].replace(0, np.nan)).round(2)

    # Revenue-at-Risk: if top N whales churned, how much revenue is lost?
    rev_at_risk = []
    sorted_ltv = df["ltv30"].sort_values(ascending=False).values
    for n_lost in [1, 5, 10, 50]:
        lost_rev = sorted_ltv[:min(n_lost, len(sorted_ltv))].sum()
        pct = lost_rev / total_rev * 100 if total_rev > 0 else 0
        rev_at_risk.append({"Whales Lost": n_lost, "Revenue Lost": lost_rev,
                            "% of Total Revenue": round(pct, 1)})
    rev_at_risk_df = pd.DataFrame(rev_at_risk)

    return df, {
        "conc_df": conc_df,
        "tier_stats": tier_stats,
        "cluster_df": cluster_df,
        "pca_df": pca_df,
        "pca_explained": pca_explained if cluster_df is not None else None,
        "early_df": early_df,
        "rev_at_risk_df": rev_at_risk_df,
        "feat_cols": feat_cols,
        "cluster_features": cluster_features if cluster_df is not None else [],
        "p99": p99, "p95": p95, "p80": p80,
        "total_rev": total_rev,
        "tier_order": tier_order,
    }, None


# ── page ───────────────────────────────────────────────────────────
st.title("🐋 Whale Segmentation")
st.markdown(
    "Profile **revenue tiers** from Mega-Whale to Non-Payer. "
    "Understand what behavioral signals separate whales from the rest — and how early we can detect them."
)

cur = get_currency_info()

# Load from registry
ds_path, ds_mtime = get_registry_path()
with st.spinner("Segmenting whale tiers and clustering…"):
    df_raw, metrics, error = compute_whale_metrics(ds_path, ds_mtime)

if error:
    st.error(f"❌ {error}")
    st.stop()

conc_df = metrics["conc_df"]
tier_stats = metrics["tier_stats"]
cluster_df = metrics["cluster_df"]
pca_df = metrics["pca_df"]
pca_explained = metrics["pca_explained"]
early_df = metrics["early_df"]
rev_at_risk_df = metrics["rev_at_risk_df"]
cluster_features = metrics["cluster_features"]
p99, p95, p80 = metrics["p99"], metrics["p95"], metrics["p80"]
total_rev = metrics["total_rev"]
tier_order = metrics["tier_order"]
n_users = len(df_raw)

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Whale_Segmentation.md", "📄 Full Whale Segmentation Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Revenue Concentration")
st.info(
    """💡 **What is Revenue Concentration?**

In most F2P games, a tiny fraction of users generate the vast majority of revenue.
This is the **whale economy** — understanding *how concentrated* your revenue is tells you
how much business risk is tied to a small number of players.

- **Healthy range:** Top 5% contributes 40–60% of revenue — diverse enough to be resilient.
- **High concentration (>70%):** Revenue depends heavily on whales — losing even a few can be catastrophic.
- **Use this data** to decide how aggressively to invest in whale retention vs. broadening the payer base.""",
    icon="📊"
)

top1_share = conc_df[conc_df["Top %"] == "Top 1%"]["Revenue Share %"].iloc[0]
top5_share = conc_df[conc_df["Top %"] == "Top 5%"]["Revenue Share %"].iloc[0]
payer_rate = (df_raw["ltv30"] > 0).mean() * 100
whale_count = (df_raw["ltv30"] >= p95).sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Top 1% Revenue Share", f"{top1_share:.1f}%", "of total LTV30")
with k2:
    st.metric("Top 5% Revenue Share", f"{top5_share:.1f}%", "of total LTV30")
with k3:
    st.metric("Payer Rate (D30)", f"{payer_rate:.1f}%")
with k4:
    st.metric("Whale Users (top 5%)", f"{whale_count:,}", f"of {n_users:,} total")

# ── Revenue Concentration Chart ─────────────────────────────────────
st.markdown("---")
st.header("📈 Revenue Concentration Curve")
col1, col2 = st.columns(2)

with col1:
    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        x=conc_df["Top %"], y=conc_df["Revenue Share %"],
        marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"],
        text=conc_df["Revenue Share %"].astype(str) + "%",
        textposition="outside",
    ))
    fig_conc.update_layout(
        title="% of Total LTV30 by Top-N% Users",
        yaxis_title="Revenue Share (%)", height=380,
        yaxis=dict(range=[0, 115]),
    )
    st.plotly_chart(fig_conc, use_container_width=True)

with col2:
    # Tier revenue pie
    tier_pie = tier_stats[["whale_tier", "ltv30_sum"]].copy()
    tier_pie = tier_pie[tier_pie["ltv30_sum"] > 0]
    fig_pie = px.pie(
        tier_pie, names="whale_tier", values="ltv30_sum",
        title="LTV30 Revenue by Tier",
        color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#95a5a6"],
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Revenue at Risk ───────────────────────────────────────────
st.markdown("---")
st.header("⚠️ Revenue at Risk")
st.markdown(
    "If your top whales churned tomorrow, how much revenue would you lose? "
    "This stress-test helps size the **retention investment** needed to protect your revenue base."
)

col_rar1, col_rar2 = st.columns(2)
with col_rar1:
    fig_rar = go.Figure()
    fig_rar.add_trace(go.Bar(
        x=rev_at_risk_df["Whales Lost"].astype(str) + " whales",
        y=rev_at_risk_df["% of Total Revenue"],
        marker_color=["#c0392b", "#e74c3c", "#e67e22", "#f1c40f"],
        text=rev_at_risk_df["% of Total Revenue"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig_rar.update_layout(
        title="Revenue Lost if Top-N Whales Churn",
        yaxis_title="% of Total Revenue", height=380,
        yaxis=dict(range=[0, rev_at_risk_df["% of Total Revenue"].max() * 1.3]),
    )
    st.plotly_chart(fig_rar, use_container_width=True)

with col_rar2:
    tbl_rar = rev_at_risk_df.copy()
    tbl_rar["Revenue Lost"] = tbl_rar["Revenue Lost"].apply(
        lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
    st.dataframe(tbl_rar, use_container_width=True, hide_index=True)
    st.markdown(
        "> **Rule of thumb:** If losing 10 whales would cost >5% of revenue, "
        "you need a dedicated **VIP retention program** with 1:1 account management. "
        "If losing 50 whales costs >20%, consider diversifying your monetization strategy."
    )

# ── Tier Behavioral Profile ───────────────────────────────────────
st.markdown("---")
st.header("🎮 Behavioral Profile by Tier")
st.info(
    """💡 **How to read this radar chart:**

Each axis represents a D7 behavioral metric, **normalized to 0–1** so all features are comparable.
A tier whose polygon covers more area has **higher engagement across all dimensions**.

- **Mega-Whales** typically dominate every axis — they play more, log in more, and progress further.
- **The gap between Mega-Whale and Whale** shows whether top spenders are also top players, or just big purchasers.
- **If Minnows have high engagement but low spend**, they are your best candidates for monetization nudges.

**Actions:** Look for tiers with high `games` and `active_days` but low `rev` — these users are engaged but under-monetized.""",
    icon="🎮"
)

feat_display = [c for c in ["games_d7_mean", "active_days_d7_mean", "win_rate_d7_mean",
                              "kd_d7_mean", "max_level_seen_d7_mean", "login_rows_d7_mean",
                              "rev_d7_mean"] if c in tier_stats.columns]

if feat_display:
    radar_data = tier_stats[["whale_tier"] + feat_display].copy()
    # Normalize each feature 0–1 for radar
    for c in feat_display:
        col_max = radar_data[c].max()
        radar_data[c] = radar_data[c] / col_max if col_max > 0 else 0

    fig_radar = go.Figure()
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#95a5a6"]
    labels = [c.replace("_d7_mean", "").replace("_mean", "") for c in feat_display]
    for i, row in radar_data.iterrows():
        tier_name = row["whale_tier"]
        if tier_name not in tier_order[:3]:
            continue
        vals = [row[c] for c in feat_display]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            fill="toself", name=str(tier_name),
            line_color=colors[tier_order.index(str(tier_name))],
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalized Behavioral Profile (top 3 tiers)",
        height=450, showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Tier stats table
st.subheader("Tier Summary Table")
tbl_cols = ["whale_tier", "ltv30_count", "rev_share_%"]
tbl_cols += [c for c in ["ltv30_mean", "games_d7_mean", "active_days_d7_mean",
                           "win_rate_d7_mean", "kd_d7_mean"] if c in tier_stats.columns]
tbl = tier_stats[tbl_cols].copy()
if "ltv30_mean" in tbl.columns:
    tbl["ltv30_mean"] = tbl["ltv30_mean"].apply(
        lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl.columns = (["Tier", "Users", "Rev Share %"] +
               [c.replace("_d7_mean", "").replace("_mean", "").replace("ltv30", "LTV30")
                for c in tbl_cols[3:]])
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Clustering ─────────────────────────────────────────────────────
if cluster_df is not None:
    st.markdown("---")
    st.header("🔵 Behavioral Clusters (K-Means, k=4)")
    st.info(
        """💡 **What are Behavioral Clusters?**

We group users into 4 clusters based **purely on how they play** — games played, active days,
win rate, K/D ratio, level progression, and login frequency. **No revenue data is used.**

The goal: **Can we predict who will spend based only on how they engage?**

- If a cluster has high engagement AND high LTV30 → these are your **natural whales** — skilled, committed players who also spend.
- If a cluster has high engagement but LOW LTV30 → these are **under-monetized enthusiasts** — prime targets for first-purchase offers.
- If a cluster has low engagement but high LTV30 → these are **impulse spenders** — they pay but don't stick around. Retention risk.

**The scatter plot below** projects all users onto 2 dimensions (via PCA) so you can visually see how distinct the clusters are.
Well-separated clusters = the algorithm found real behavioral patterns. Overlapping clusters = user behaviors are more homogeneous.""",
        icon="🔵"
    )

    # PCA scatter visualization
    if pca_df is not None:
        col_pca1, col_pca2 = st.columns(2)
        cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        with col_pca1:
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="cluster_label",
                color_discrete_sequence=cluster_colors,
                opacity=0.5,
                title=f"User Clusters in 2D (PCA — {sum(pca_explained)*100:.0f}% variance explained)",
                labels={"PC1": f"PC1 ({pca_explained[0]*100:.0f}%)",
                        "PC2": f"PC2 ({pca_explained[1]*100:.0f}%)"},
                height=420,
            )
            fig_pca.update_traces(marker=dict(size=3))
            fig_pca.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_pca2:
            fig_cluster = go.Figure()
            for i, row in cluster_df.iterrows():
                fig_cluster.add_trace(go.Bar(
                    x=[row["cluster_label"]],
                    y=[convert_vnd(row["avg_ltv30"], cur["code"])],
                    name=row["cluster_label"],
                    marker_color=cluster_colors[i % len(cluster_colors)],
                    text=[format_currency(convert_vnd(row["avg_ltv30"], cur["code"]), cur["code"])],
                    textposition="outside",
                ))
            fig_cluster.update_layout(
                title=f"Avg LTV30 by Engagement Cluster ({cur['symbol']})",
                yaxis_title=f"Avg LTV30 ({cur['symbol']})", height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        fig_cluster = go.Figure()
        cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        for i, row in cluster_df.iterrows():
            fig_cluster.add_trace(go.Bar(
                x=[row["cluster_label"]],
                y=[convert_vnd(row["avg_ltv30"], cur["code"])],
                name=row["cluster_label"],
                marker_color=cluster_colors[i % len(cluster_colors)],
                text=[format_currency(convert_vnd(row["avg_ltv30"], cur["code"]), cur["code"])],
                textposition="outside",
            ))
        fig_cluster.update_layout(
            title=f"Avg LTV30 by Engagement Cluster ({cur['symbol']})",
            yaxis_title=f"Avg LTV30 ({cur['symbol']})", height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # Enhanced cluster table
    tbl_cols_cl = ["cluster_label", "users", "avg_ltv30", "payer_rate", "whale_rate", "rev_share_%"]
    cluster_tbl = cluster_df[[c for c in tbl_cols_cl if c in cluster_df.columns]].copy()
    cluster_tbl["avg_ltv30"] = cluster_tbl["avg_ltv30"].apply(
        lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
    cluster_tbl["payer_rate"] = (cluster_tbl["payer_rate"] * 100).round(1).astype(str) + "%"
    if "whale_rate" in cluster_tbl.columns:
        cluster_tbl["whale_rate"] = (cluster_tbl["whale_rate"] * 100).round(2).astype(str) + "%"
    col_names = ["Cluster", "Users", f"Avg LTV30 ({cur['symbol']})", "Payer Rate", "Whale Rate", "Rev Share %"]
    cluster_tbl.columns = col_names[:len(cluster_tbl.columns)]
    st.dataframe(cluster_tbl, use_container_width=True, hide_index=True)

# ── Early Detection ─────────────────────────────────────────
if early_df is not None:
    st.markdown("---")
    st.header("⚡ Early Whale Detection Signals")
    st.info(
        """💡 **Can we identify whales before they spend?**

This chart compares the **average D7 behavior** of users who became whales (top 5% LTV30) vs. everyone else.
The **Ratio column** is the key metric — it shows how many times higher the whale average is.

- **Ratio > 2×:** Strong early signal — whales are dramatically different on this metric.
- **Ratio ~ 1×:** Not useful for early detection — whales look similar to non-whales.
- **`first_charge_day_offset` near 0:** Whales tend to pay on Day 0 — they arrive with purchase intent.
- **`games_d7` and `active_days_d7` > 2×:** Whales are also power players, not just big spenders.

**Action:** Build a D1 whale scoring rule: if `games_d7 > X` AND `first_charge_day = 0`, flag for VIP treatment.""",
        icon="⚡"
    )

    col_early1, col_early2 = st.columns(2)
    with col_early1:
        fig_early = go.Figure()
        fig_early.add_trace(go.Bar(
            name="Non-Whale", x=early_df["Feature"],
            y=early_df["Non-Whale"], marker_color="#95a5a6",
        ))
        fig_early.add_trace(go.Bar(
            name="Whale (top 5%)", x=early_df["Feature"],
            y=early_df["Whale"], marker_color="#e74c3c",
        ))
        fig_early.update_layout(
            title="Whale vs Non-Whale: Average Feature Values",
            barmode="group", height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_early, use_container_width=True)

    with col_early2:
        # Ratio bar chart — more intuitive for business users
        ratio_df = early_df.copy()
        ratio_df = ratio_df.sort_values("Whale/Non-Whale Ratio", ascending=True)
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Bar(
            x=ratio_df["Whale/Non-Whale Ratio"],
            y=ratio_df["Feature"],
            orientation="h",
            marker_color=ratio_df["Whale/Non-Whale Ratio"].apply(
                lambda v: "#e74c3c" if v >= 2 else "#e67e22" if v >= 1.5 else "#95a5a6"),
            text=ratio_df["Whale/Non-Whale Ratio"].apply(lambda v: f"{v:.1f}×"),
            textposition="outside",
        ))
        fig_ratio.add_vline(x=1.0, line_dash="dash", line_color="gray",
                            annotation_text="1× (no difference)")
        fig_ratio.update_layout(
            title="Whale ÷ Non-Whale Ratio (higher = stronger signal)",
            xaxis_title="Ratio", height=420,
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.dataframe(
        early_df.rename(columns={"Whale/Non-Whale Ratio": "Ratio (Whale÷Non-Whale)"}),
        use_container_width=True, hide_index=True,
    )

# ── Insights & Playbook ───────────────────────────────────────
st.markdown("---")
st.header("💡 Key Findings")

find_col1, find_col2 = st.columns(2)
with find_col1:
    st.markdown(f"""
**Revenue Structure:**
- Top 1% of users = **{top1_share:.1f}%** of revenue
- Top 5% of users = **{top5_share:.1f}%** of revenue
- Whale threshold (top 5%): **{format_currency(convert_vnd(p95, cur['code']), cur['code'])}** LTV30
- Mega-whale threshold (top 1%): **{format_currency(convert_vnd(p99, cur['code']), cur['code'])}** LTV30
""")
with find_col2:
    st.markdown(f"""
**Behavioral Signals:**
- Whales play significantly more games and log in more days in D7
- Same-day charge (`first_charge_day_offset = 0`) is the strongest whale signal
- Engagement clusters can predict spending potential before any purchase
- Payer rate: **{payer_rate:.1f}%** — most users never pay, but those who do are highly valuable
""")

st.markdown("### 🎯 Whale Management Playbook")
play_col1, play_col2, play_col3 = st.columns(3)
with play_col1:
    st.markdown("""
**🔍 Identify (D0–D1)**
- Flag users with D0 purchase + high games count as potential whales
- Use engagement cluster assignment as a real-time scoring signal
- Prioritize D0 converters for VIP onboarding flow
""")
with play_col2:
    st.markdown("""
**💰 Monetize (D1–D7)**
- For engaged non-payers in whale-like clusters: trigger first-purchase offer
- For D0 payers: offer bundle/subscription by D3
- Use whale cluster as **UA lookalike seed** instead of broad payer seed
""")
with play_col3:
    st.markdown("""
**🛡️ Retain (D7+)**
- Set up whale churn alerts (see **Churn Prediction** page)
- Assign top 50 whales to VIP account management
- Monitor weekly: if a whale\'s active_days drops, trigger retention offer
""")

st.markdown(
    "> **Cross-reference this page with:** "
    "[Time-to-First-Purchase](#) for conversion timing, "
    "[Channel x Whale Quality](#) for acquisition source, "
    "[Churn Prediction](#) for retention risk."
)
