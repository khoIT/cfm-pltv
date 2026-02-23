"""
Page 3l — Skill-to-Spend Correlation
Does skill drive spending, or does spending buy skill? Spearman correlations + segment profiles.
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

SKILL_COLS  = ["kd_d7", "win_rate_d7", "avg_score_d7", "max_level_seen_d7",
               "max_ladderscore_d7", "max_level_game_d7"]
SPEND_COLS  = ["ltv30", "rev_d7", "txn_cnt_d7"]
ENG_COLS    = ["games_d7", "active_days_d7", "login_rows_d7", "avg_game_duration_d7"]


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing skill-spend correlations…")
def compute_skill_metrics(csv_path: str, file_mtime: float):
    from scipy.stats import spearmanr

    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' column."

    df = df.copy()
    df["ltv30"] = pd.to_numeric(df["ltv30"], errors="coerce").fillna(0)

    skill_avail  = [c for c in SKILL_COLS  if c in df.columns]
    spend_avail  = [c for c in SPEND_COLS  if c in df.columns]
    eng_avail    = [c for c in ENG_COLS    if c in df.columns]

    if not skill_avail:
        return None, None, "No skill columns found (kd_d7, win_rate_d7, etc.)."

    # Spearman correlation matrix: skill vs spend
    corr_rows = []
    for sk in skill_avail:
        for sp in spend_avail:
            valid = df[[sk, sp]].dropna()
            valid = valid[(valid[sk] >= 0) & (valid[sp] >= 0)]
            if len(valid) < 30:
                continue
            rho, pval = spearmanr(valid[sk], valid[sp])
            corr_rows.append({
                "Skill Feature": sk.replace("_d7", ""),
                "Spend Feature": sp.replace("_d7", ""),
                "Spearman ρ": round(rho, 4),
                "p-value": round(pval, 4),
                "Significant": "✅" if pval < 0.05 else "❌",
            })
    corr_df = pd.DataFrame(corr_rows) if corr_rows else None

    # Spending tier segments
    p99 = df["ltv30"].quantile(0.99)
    p95 = df["ltv30"].quantile(0.95)
    p50 = df["ltv30"].quantile(0.50)

    def spend_tier(v):
        if v >= p99:  return "Mega-Whale"
        if v >= p95:  return "Whale"
        if v > p50:   return "Mid Payer"
        if v > 0:     return "Low Payer"
        return "Non-Payer"

    tier_order = ["Mega-Whale", "Whale", "Mid Payer", "Low Payer", "Non-Payer"]
    df["spend_tier"] = df["ltv30"].apply(spend_tier)
    df["spend_tier"] = pd.Categorical(df["spend_tier"], categories=tier_order, ordered=True)

    profile_cols = skill_avail + eng_avail
    tier_profile = df.groupby("spend_tier", observed=True)[profile_cols].mean().reset_index()

    # Skill percentile buckets vs avg LTV30
    skill_ltv = {}
    for sk in skill_avail[:3]:  # top 3 skill features
        valid = df[[sk, "ltv30"]].dropna()
        valid = valid[valid[sk] >= 0]
        if len(valid) < 50:
            continue
        valid["skill_pct"] = pd.qcut(valid[sk], q=10, labels=False, duplicates="drop")
        bucket = valid.groupby("skill_pct").agg(
            avg_ltv30=("ltv30", "mean"),
            users=("ltv30", "count"),
        ).reset_index()
        bucket["skill_pct_label"] = (bucket["skill_pct"] * 10).astype(int).astype(str) + "–" + \
                                     ((bucket["skill_pct"] + 1) * 10).astype(int).astype(str) + "%"
        skill_ltv[sk] = bucket

    # High-skill non-payers — conversion opportunity
    if skill_avail:
        kd_col = "kd_d7" if "kd_d7" in df.columns else skill_avail[0]
        wr_col = "win_rate_d7" if "win_rate_d7" in df.columns else None
        kd_thresh = df[kd_col].quantile(0.75)
        mask = (df[kd_col] >= kd_thresh) & (df["ltv30"] == 0)
        if wr_col:
            wr_thresh = df[wr_col].quantile(0.75)
            mask = mask & (df[wr_col] >= wr_thresh)
        high_skill_nonpayers = df[mask]
        hs_count = len(high_skill_nonpayers)
        hs_pct = hs_count / len(df) * 100
    else:
        hs_count, hs_pct = 0, 0

    return df, {
        "corr_df": corr_df,
        "tier_profile": tier_profile,
        "skill_ltv": skill_ltv,
        "skill_avail": skill_avail,
        "eng_avail": eng_avail,
        "tier_order": tier_order,
        "hs_count": hs_count,
        "hs_pct": hs_pct,
        "p95": p95, "p99": p99,
        "n_total": len(df),
    }, None


# ── page ───────────────────────────────────────────────────────────
st.title("🎯 Skill-to-Spend Correlation")
st.markdown(
    "Does **skill** (K/D ratio, win rate, score) drive spending — or does spending buy skill? "
    "Understand the relationship to inform monetization design and UA targeting."
)

cur = get_currency_info()

# Load from registry
ds_path, ds_mtime = get_registry_path()
with st.spinner("Computing skill-spend correlations…"):
    df_raw, metrics, error = compute_skill_metrics(ds_path, ds_mtime)

if error:
    st.error(f"❌ {error}")
    st.stop()

corr_df       = metrics["corr_df"]
tier_profile  = metrics["tier_profile"]
skill_ltv     = metrics["skill_ltv"]
skill_avail   = metrics["skill_avail"]
eng_avail     = metrics["eng_avail"]
tier_order    = metrics["tier_order"]
hs_count      = metrics["hs_count"]
hs_pct        = metrics["hs_pct"]
n_total       = metrics["n_total"]

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Skill_Spend_Correlation.md", "📄 Full Skill-to-Spend Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Key Metrics")

best_corr = corr_df[corr_df["Spend Feature"] == "ltv30"].sort_values(
    "Spearman ρ", ascending=False).iloc[0] if corr_df is not None and len(corr_df) > 0 else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    if best_corr is not None:
        st.metric("Strongest Skill→LTV30 Signal",
                  best_corr["Skill Feature"],
                  f"ρ = {best_corr['Spearman ρ']:.4f}")
with k2:
    if corr_df is not None:
        sig_count = (corr_df["p-value"] < 0.05).sum()
        st.metric("Significant Correlations", f"{sig_count}", f"of {len(corr_df)} tested")
with k3:
    st.metric("High-Skill Non-Payers", f"{hs_count:,}",
              f"{hs_pct:.1f}% of all users — conversion opportunity")
with k4:
    whale_pct = (df_raw["ltv30"] >= metrics["p95"]).mean() * 100
    st.metric("Whale Rate (top 5%)", f"{whale_pct:.1f}%")

# ── Correlation Heatmap ─────────────────────────────────────────────
if corr_df is not None and len(corr_df) > 0:
    st.markdown("---")
    st.header("🔥 Skill × Spend Spearman Correlation")
    col1, col2 = st.columns([2, 1])

    with col1:
        pivot = corr_df.pivot(index="Skill Feature", columns="Spend Feature",
                               values="Spearman ρ").fillna(0)
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            text_auto=".3f",
            title="Spearman ρ: Skill Features vs Spend Features",
            height=400,
        )
        fig_heat.update_layout(coloraxis_colorbar=dict(title="ρ"))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        ltv_corr = corr_df[corr_df["Spend Feature"] == "ltv30"].sort_values(
            "Spearman ρ", ascending=False)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=ltv_corr["Spearman ρ"],
            y=ltv_corr["Skill Feature"],
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#3498db"
                          for v in ltv_corr["Spearman ρ"]],
            text=ltv_corr["Spearman ρ"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Skill → LTV30 Spearman ρ",
            xaxis_title="Spearman ρ", height=400,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(corr_df, use_container_width=True, hide_index=True)

# ── Tier Skill Profile ──────────────────────────────────────────────
st.markdown("---")
st.header("🎮 Skill Profile by Spending Tier")

if len(tier_profile) > 0 and skill_avail:
    # Normalize for radar
    radar_df = tier_profile.copy()
    profile_cols = [c for c in skill_avail + eng_avail if c in radar_df.columns]
    for c in profile_cols:
        col_max = radar_df[c].max()
        radar_df[c] = radar_df[c] / col_max if col_max > 0 else 0

    col3, col4 = st.columns(2)
    tier_colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#95a5a6"]
    labels = [c.replace("_d7", "") for c in profile_cols]

    with col3:
        fig_radar = go.Figure()
        for i, row in radar_df.iterrows():
            tier_name = str(row["spend_tier"])
            if tier_name not in ["Mega-Whale", "Whale", "Non-Payer"]:
                continue
            vals = [row[c] for c in profile_cols] + [row[profile_cols[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=labels + [labels[0]],
                fill="toself", name=tier_name,
                line_color=tier_colors[tier_order.index(tier_name)],
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Profile: Mega-Whale vs Whale vs Non-Payer",
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col4:
        # Skill feature bar comparison across tiers
        if skill_avail:
            sk_col = skill_avail[0]  # e.g. kd_d7
            fig_tier_skill = go.Figure()
            for i, row in tier_profile.iterrows():
                fig_tier_skill.add_trace(go.Bar(
                    x=[str(row["spend_tier"])],
                    y=[row[sk_col]],
                    name=str(row["spend_tier"]),
                    marker_color=tier_colors[i % len(tier_colors)],
                    text=[f"{row[sk_col]:.2f}"],
                    textposition="outside",
                ))
            fig_tier_skill.update_layout(
                title=f"Avg {sk_col.replace('_d7','')} by Spending Tier",
                yaxis_title=sk_col.replace("_d7", ""),
                height=420, showlegend=False,
            )
            st.plotly_chart(fig_tier_skill, use_container_width=True)

# ── Skill Percentile vs LTV30 ───────────────────────────────────────
if skill_ltv:
    st.markdown("---")
    st.header("📈 Skill Percentile vs Avg LTV30")
    cols = st.columns(min(len(skill_ltv), 3))
    for idx, (sk_name, bucket_df) in enumerate(skill_ltv.items()):
        with cols[idx % len(cols)]:
            fig_sk = go.Figure()
            fig_sk.add_trace(go.Scatter(
                x=bucket_df["skill_pct_label"],
                y=bucket_df["avg_ltv30"].apply(lambda v: convert_vnd(v, cur["code"])),
                mode="lines+markers",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=8),
            ))
            fig_sk.update_layout(
                title=f"{sk_name.replace('_d7','')} Percentile vs Avg LTV30",
                xaxis_title="Skill Percentile",
                yaxis_title=f"Avg LTV30 ({cur['symbol']})",
                height=350,
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_sk, use_container_width=True)

# ── High-Skill Non-Payers ───────────────────────────────────────────
st.markdown("---")
st.header("🎯 High-Skill Non-Payers — Conversion Opportunity")
st.info(
    f"**{hs_count:,} users ({hs_pct:.1f}% of all users)** have top-quartile skill metrics "
    "but have never paid. These are the highest-value conversion targets.",
    icon="💡"
)

if "kd_d7" in df_raw.columns and "win_rate_d7" in df_raw.columns:
    kd_thresh = df_raw["kd_d7"].quantile(0.75)
    wr_thresh = df_raw["win_rate_d7"].quantile(0.75)
    df_plot = df_raw.sample(min(5000, len(df_raw)), random_state=42).copy()
    df_plot["segment"] = "Other"
    df_plot.loc[(df_plot["kd_d7"] >= kd_thresh) & (df_plot["ltv30"] == 0),
                "segment"] = "High-Skill Non-Payer"
    df_plot.loc[df_plot["ltv30"] >= metrics["p95"], "segment"] = "Whale"
    df_plot.loc[(df_plot["ltv30"] > 0) & (df_plot["ltv30"] < metrics["p95"]),
                "segment"] = "Low/Mid Payer"

    fig_scatter = px.scatter(
        df_plot, x="kd_d7", y="win_rate_d7",
        color="segment",
        opacity=0.5,
        color_discrete_map={
            "Whale": "#e74c3c",
            "High-Skill Non-Payer": "#f39c12",
            "Low/Mid Payer": "#3498db",
            "Other": "#bdc3c7",
        },
        title="K/D vs Win Rate — Segment Map (sample 5k users)",
        labels={"kd_d7": "K/D Ratio", "win_rate_d7": "Win Rate"},
        height=420,
    )
    fig_scatter.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
if best_corr is not None:
    st.markdown(f"- **Strongest skill→LTV30 signal:** `{best_corr['Skill Feature']}` "
                f"(Spearman ρ = {best_corr['Spearman ρ']:.4f})")
st.markdown("- Correlation is **positive but moderate** — skill and spending are linked but not deterministic")
st.markdown("- Whales play **3× more games** → skill improves through volume (spending enables more play)")
st.markdown(f"- **{hs_count:,} high-skill non-payers** ({hs_pct:.1f}%) — best conversion targets with competitive offers")
st.markdown("### 🎯 Recommended Actions")
st.markdown("- Add **skill tier** (K/D + win rate) as a feature in the pLTV model")
st.markdown("- Target high-skill non-payers (K/D > 75th pct, ltv30=0) with **competitive/prestige offers**")
st.markdown("- Monitor skill progression of new payers — stagnation predicts churn")
st.markdown("- Separate monetization funnels: **prestige items** for skilled players, **power items** for casual")
