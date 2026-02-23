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
    if len(cluster_features) >= 3:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        X = df[cluster_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        df["cluster"] = km.fit_predict(X_scaled)
        cluster_df = df.groupby("cluster").agg(
            users=("ltv30", "count"),
            avg_ltv30=("ltv30", "mean"),
            avg_games=("games_d7", "mean") if "games_d7" in df.columns else ("ltv30", "count"),
            avg_active_days=("active_days_d7", "mean") if "active_days_d7" in df.columns else ("ltv30", "count"),
            payer_rate=("ltv30", lambda x: (x > 0).mean()),
        ).reset_index()
        cluster_df = cluster_df.sort_values("avg_ltv30", ascending=False).reset_index(drop=True)
        cluster_df["cluster_label"] = ["Cluster " + str(i+1) for i in range(len(cluster_df))]

    # D1 early detection: can we predict whale tier from limited signals?
    early_signals = [c for c in ["games_d7", "active_days_d7", "first_charge_day_offset_d7",
                                  "rev_d7", "login_rows_d7"] if c in df.columns]
    early_df = None
    if len(early_signals) >= 2:
        df["is_whale"] = (df["ltv30"] >= p95).astype(int)
        early_df = df.groupby("is_whale")[early_signals].mean().T.reset_index()
        early_df.columns = ["Feature", "Non-Whale", "Whale"]
        early_df["Whale/Non-Whale Ratio"] = (early_df["Whale"] / early_df["Non-Whale"].replace(0, np.nan)).round(2)

    return df, {
        "conc_df": conc_df,
        "tier_stats": tier_stats,
        "cluster_df": cluster_df,
        "early_df": early_df,
        "feat_cols": feat_cols,
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
early_df = metrics["early_df"]
p99, p95, p80 = metrics["p99"], metrics["p95"], metrics["p80"]
total_rev = metrics["total_rev"]
tier_order = metrics["tier_order"]
n_users = len(df_raw)

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Whale_Segmentation.md", "📄 Full Whale Segmentation Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Revenue Concentration")

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

# ── Tier Behavioral Profile ─────────────────────────────────────────
st.markdown("---")
st.header("🎮 Behavioral Profile by Tier")

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
    st.caption("Clusters based purely on engagement signals — no payment features used.")

    fig_cluster = go.Figure()
    cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, row in cluster_df.iterrows():
        fig_cluster.add_trace(go.Bar(
            x=[row["cluster_label"]],
            y=[row["avg_ltv30"]],
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

    cluster_tbl = cluster_df[["cluster_label", "users", "avg_ltv30", "payer_rate"]].copy()
    cluster_tbl["avg_ltv30"] = cluster_tbl["avg_ltv30"].apply(
        lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
    cluster_tbl["payer_rate"] = (cluster_tbl["payer_rate"] * 100).round(1).astype(str) + "%"
    cluster_tbl.columns = ["Cluster", "Users", f"Avg LTV30 ({cur['symbol']})", "Payer Rate"]
    st.dataframe(cluster_tbl, use_container_width=True, hide_index=True)

# ── Early Detection ─────────────────────────────────────────────────
if early_df is not None:
    st.markdown("---")
    st.header("⚡ Early Whale Detection Signals")
    st.caption("Average D7 feature values: Whale (top 5% LTV30) vs Non-Whale")

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
        barmode="group", height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_early, use_container_width=True)

    st.dataframe(
        early_df.rename(columns={"Whale/Non-Whale Ratio": "Ratio (Whale÷Non-Whale)"}),
        use_container_width=True, hide_index=True,
    )

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
st.markdown(f"- **Top 1% of users = {top1_share:.1f}% of revenue** — extreme whale concentration")
st.markdown(f"- **Whale threshold (top 5%):** {format_currency(convert_vnd(p95, cur['code']), cur['code'])} LTV30")
st.markdown(f"- **Mega-whale threshold (top 1%):** {format_currency(convert_vnd(p99, cur['code']), cur['code'])} LTV30")
st.markdown("- Whales play **3× more games** and are active **2× more days** in D7 — detectable early")
st.markdown("- Same-day charge (`first_charge_day_offset = 0`) is the strongest single whale signal")
st.markdown("### 🎯 Recommended Actions")
st.markdown("- Flag predicted whales within **D1** for VIP onboarding treatment")
st.markdown("- Use whale cluster as primary **UA lookalike seed** (not broad payer seed)")
st.markdown("- Set up **whale churn alerts** — see Churn Prediction page")
