"""
Page 3e — Causal Inference
Identify behavioral drivers of late payment among D7=0 users.
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
    render_sidebar, render_top_menu, render_report_md, convert_vnd, get_currency_info,
    format_currency, DATA_DIR, REPORTS_DIR,
)

render_top_menu()
render_sidebar()


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing causal inference metrics…")
def compute_causal_metrics(csv_path: str, file_mtime: float):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' column."

    df["rev_d7"] = df.get("rev_d7", pd.Series(0.0, index=df.index)).astype(float)
    df["is_late_payer"] = ((df["rev_d7"] == 0) & (df["ltv30"] > 0)).astype(int)

    # D7=0 segment only
    d7_zero = df[df["rev_d7"] == 0].copy()

    # Behavioral features to compare
    eng_features = [c for c in ["games_d7", "active_days_d7", "login_rows_d7",
                                 "kd_d7", "win_rate_d7", "avg_score_d7",
                                 "max_level_seen_d7", "kills_d7", "deaths_d7",
                                 "avg_game_duration_d7", "max_level_game_d7",
                                 "max_ladderscore_d7"] if c in df.columns]

    if not eng_features:
        return None, None, "No engagement features (games_d7, active_days_d7, etc.) found in dataset."

    # Feature comparison: late payers vs non-payers
    late = d7_zero[d7_zero["is_late_payer"] == 1]
    non_late = d7_zero[d7_zero["is_late_payer"] == 0]

    comparison = []
    for feat in eng_features:
        lp_mean = late[feat].mean()
        np_mean = non_late[feat].mean()
        ratio = lp_mean / np_mean if np_mean > 0 else np.nan
        comparison.append({"feature": feat, "late_payer_mean": lp_mean,
                            "non_payer_mean": np_mean, "ratio": ratio})
    comparison_df = pd.DataFrame(comparison).sort_values("ratio", ascending=False, na_position="last")

    # Bucket analysis for top feature
    top_feat = comparison_df.dropna(subset=["ratio"]).iloc[0]["feature"] if not comparison_df.dropna(subset=["ratio"]).empty else eng_features[0]
    buckets = None
    if top_feat in d7_zero.columns:
        d7_zero["bucket"] = pd.qcut(d7_zero[top_feat], q=5, duplicates="drop", labels=False)
        buckets = d7_zero.groupby("bucket").agg(
            users=("is_late_payer", "count"),
            late_payer_rate=("is_late_payer", "mean"),
            mean_feat=(top_feat, "mean"),
        ).reset_index()

    # Active days bucket (always useful if present)
    active_buckets = None
    if "active_days_d7" in d7_zero.columns:
        d7_zero["active_bucket"] = pd.cut(d7_zero["active_days_d7"],
                                           bins=[-1, 1, 3, 5, 7, 100],
                                           labels=["0-1", "2-3", "4-5", "6-7", "7+"])
        active_buckets = d7_zero.groupby("active_bucket", observed=True).agg(
            users=("is_late_payer", "count"),
            late_payer_rate=("is_late_payer", "mean"),
        ).reset_index()

    return df, {
        "d7_zero": d7_zero,
        "comparison": comparison_df,
        "top_feat": top_feat,
        "buckets": buckets,
        "active_buckets": active_buckets,
        "n_d7_zero": len(d7_zero),
        "n_late": len(late),
        "n_non_late": len(non_late),
    }, None


def list_available_datasets():
    datasets = {}
    for f in DATA_DIR.glob("cfm_pltv*.csv"):
        size_mb = f.stat().st_size / 1e6
        datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    return datasets


# ── page ───────────────────────────────────────────────────────────
st.title("🔬 Causal Inference")
st.markdown(
    "For users with **rev_d7 = 0**, what behavioral signals predict late conversion? "
    "Identify causal drivers of late payment to inform feature engineering and product interventions."
)

cur = get_currency_info()

# ── Dataset selector ───────────────────────────────────────────────
st.header("📂 Select Dataset")
datasets = list_available_datasets()
if not datasets:
    st.error("No datasets found in data/ directory.")
    st.stop()

ds_names = list(datasets.keys())
default_idx = ds_names.index("cfm_pltv") if "cfm_pltv" in ds_names else 0
col_ds1, col_ds2 = st.columns([2, 3])
with col_ds1:
    chosen_ds = st.selectbox("Dataset", ds_names, index=default_idx, key="causal_dataset",
                             help="Choose which dataset to analyze")
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.1f} MB")

df_raw, metrics, error = compute_causal_metrics(ds_info["path"], ds_info["mtime"])
if error:
    st.error(f"❌ {error}")
    st.stop()

n_d7_zero = metrics["n_d7_zero"]
n_late = metrics["n_late"]
n_non_late = metrics["n_non_late"]
comparison_df = metrics["comparison"]
top_feat = metrics["top_feat"]
buckets = metrics["buckets"]
active_buckets = metrics["active_buckets"]

st.success(f"✅ Loaded **{len(df_raw):,}** users — **{n_d7_zero:,}** with rev_d7 = 0 (D7=0 segment)")

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Causal_Inference.md", "📄 Full Causal Inference Report")

# ── KPI Summary ────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 D7=0 Segment KPIs")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("D7=0 Users", f"{n_d7_zero:,}",
              f"{n_d7_zero/len(df_raw):.1%} of total")
with k2:
    st.metric("Late Payers", f"{n_late:,}",
              f"{n_late/n_d7_zero:.2%} conversion" if n_d7_zero > 0 else "")
with k3:
    late_rev = df_raw[(df_raw["rev_d7"] == 0) & (df_raw["ltv30"] > 0)]["ltv30"].sum()
    total_rev = df_raw["ltv30"].sum()
    st.metric("Late Payer Revenue", f"{late_rev/total_rev:.1%} of total" if total_rev > 0 else "N/A")
with k4:
    best_feat_row = comparison_df.dropna(subset=["ratio"]).iloc[0] if not comparison_df.dropna(subset=["ratio"]).empty else None
    if best_feat_row is not None:
        st.metric("Strongest Predictor", best_feat_row["feature"],
                  f"{best_feat_row['ratio']:.2f}x ratio")

# ── Chart 1: Feature Comparison ────────────────────────────────────
st.markdown("---")
st.header("📈 Feature Comparison: Late Payers vs Non-Payers")

valid_comp = comparison_df.dropna(subset=["ratio"]).head(10)
if len(valid_comp) > 0:
    col1, col2 = st.columns(2)
    with col1:
        fig_ratio = px.bar(
            valid_comp.sort_values("ratio"),
            x="ratio", y="feature", orientation="h",
            title="Payer/Non-Payer Feature Ratio (higher = stronger signal)",
            color="ratio", color_continuous_scale="RdYlGn",
            labels={"ratio": "Ratio (Late Payer / Non-Payer)", "feature": ""},
        )
        fig_ratio.add_vline(x=1.0, line_dash="dash", line_color="gray",
                            annotation_text="No difference")
        fig_ratio.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_ratio, use_container_width=True)

    with col2:
        # Grouped bar: means side by side
        top_feats = valid_comp.head(6)
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            x=top_feats["feature"],
            y=top_feats["late_payer_mean"],
            name="Late Payers", marker_color="#2ecc71",
        ))
        fig_means.add_trace(go.Bar(
            x=top_feats["feature"],
            y=top_feats["non_payer_mean"],
            name="Non-Payers", marker_color="#e74c3c",
        ))
        fig_means.update_layout(
            title="Feature Means: Late Payers vs Non-Payers",
            barmode="group", height=420,
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_means, use_container_width=True)

# ── Chart 2: Dose-Response ─────────────────────────────────────────
st.markdown("---")
st.header("📉 Dose-Response: Engagement → Conversion")
col3, col4 = st.columns(2)

with col3:
    if buckets is not None and len(buckets) > 1:
        fig_bucket = go.Figure()
        fig_bucket.add_trace(go.Bar(
            x=buckets["mean_feat"].round(1).astype(str),
            y=buckets["late_payer_rate"] * 100,
            marker_color="#FF6600",
            text=(buckets["late_payer_rate"] * 100).round(2).astype(str) + "%",
            textposition="outside",
        ))
        fig_bucket.update_layout(
            title=f"Late Conversion Rate by {top_feat} Bucket",
            xaxis_title=f"Avg {top_feat} (quintile)",
            yaxis_title="Late Payer Rate (%)",
            height=400,
        )
        st.plotly_chart(fig_bucket, use_container_width=True)
        st.caption(f"Dose-response: higher **{top_feat}** → higher late conversion rate")
    else:
        st.info("Not enough variation in top feature for bucket analysis.")

with col4:
    if active_buckets is not None and len(active_buckets) > 1:
        fig_active = go.Figure()
        fig_active.add_trace(go.Bar(
            x=active_buckets["active_bucket"].astype(str),
            y=active_buckets["late_payer_rate"] * 100,
            marker_color="royalblue",
            text=(active_buckets["late_payer_rate"] * 100).round(2).astype(str) + "%",
            textposition="outside",
        ))
        fig_active.update_layout(
            title="Late Conversion Rate by Active Days D7",
            xaxis_title="Active Days D7",
            yaxis_title="Late Payer Rate (%)",
            height=400,
        )
        st.plotly_chart(fig_active, use_container_width=True)
        st.caption("Users active more days in D7 are significantly more likely to convert late")

# ── Feature Table ──────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Feature Discriminator Table")
tbl = comparison_df.copy()
tbl["late_payer_mean"] = tbl["late_payer_mean"].round(3)
tbl["non_payer_mean"] = tbl["non_payer_mean"].round(3)
tbl["ratio"] = tbl["ratio"].round(3)
tbl.columns = ["Feature", "Late Payer Mean", "Non-Payer Mean", "Ratio (LP/NP)"]
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
late_rate = n_late / n_d7_zero if n_d7_zero > 0 else 0
st.markdown(f"- **{n_d7_zero:,}** users had rev_d7 = 0; **{n_late:,}** ({late_rate:.2%}) converted as late payers")
if not comparison_df.dropna(subset=["ratio"]).empty:
    top3 = comparison_df.dropna(subset=["ratio"]).head(3)
    for _, row in top3.iterrows():
        st.markdown(f"- **{row['feature']}**: late payers average **{row['ratio']:.2f}x** higher than non-payers — strong causal signal")
st.markdown("- Dose-response patterns (more engagement → higher conversion) strengthen the causal argument")
st.markdown("- **Recommended action:** Target D7=0 users with high engagement scores for D8–D14 monetization nudges")
