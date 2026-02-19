"""
Page 3f — Seed Optimization
Compare seed strategies for UA lookalike expansion.
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
@st.cache_data(show_spinner="Computing seed strategies…")
def compute_seed_metrics(csv_path: str, file_mtime: float, top_late_pct: float):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' column."

    df["rev_d7"] = df.get("rev_d7", pd.Series(0.0, index=df.index)).astype(float)
    df["is_payer_30"] = (df["ltv30"] > 0).astype(int)
    df["is_late_payer"] = ((df["rev_d7"] == 0) & (df["ltv30"] > 0)).astype(int)

    # Engagement score as proxy for ML prediction
    eng_cols = [c for c in ["games_d7", "active_days_d7", "login_rows_d7"] if c in df.columns]
    if eng_cols:
        score_parts = []
        for col in eng_cols:
            mx = df[col].max()
            score_parts.append(df[col] / mx if mx > 0 else 0)
        df["engagement_score"] = sum(score_parts) / len(score_parts)
    else:
        df["engagement_score"] = 0.0

    # Define seed strategies
    d7_payers = df[df["rev_d7"] > 0]
    d7_zero = df[df["rev_d7"] == 0]

    # Top N% late payers by engagement score
    threshold = d7_zero["engagement_score"].quantile(1 - top_late_pct / 100)
    predicted_late = d7_zero[d7_zero["engagement_score"] >= threshold]

    enriched = pd.concat([d7_payers, predicted_late], ignore_index=True)
    top10_rev = df.nlargest(int(len(df) * 0.10), "rev_d7")
    oracle = df[df["is_payer_30"] == 1]

    total_rev = df["ltv30"].sum()
    whale_threshold = df["ltv30"].quantile(0.90)
    total_whales = (df["ltv30"] >= whale_threshold).sum()

    def seed_stats(seed_df, name):
        n = len(seed_df)
        avg_ltv = seed_df["ltv30"].mean() if n > 0 else 0
        payer_rate = seed_df["is_payer_30"].mean() if n > 0 else 0
        whales = (seed_df["ltv30"] >= whale_threshold).sum()
        whale_capture = whales / total_whales if total_whales > 0 else 0
        total = seed_df["ltv30"].sum()
        return {
            "Strategy": name, "Seed Size": n,
            "Avg LTV30": avg_ltv, "Payer Rate": payer_rate,
            "Whale Capture": whale_capture, "Total Revenue": total,
        }

    strategies = pd.DataFrame([
        seed_stats(d7_payers, "D7 Payers Only"),
        seed_stats(enriched, f"D7 Payers + Top {top_late_pct:.0f}% Late"),
        seed_stats(top10_rev, "Top 10% by rev_d7"),
        seed_stats(oracle, "D30 Payers (Oracle)"),
    ])

    # Revenue composition of enriched seed
    enriched_from_d7 = d7_payers["ltv30"].sum()
    enriched_from_late = predicted_late["ltv30"].sum()

    return df, {
        "strategies": strategies,
        "d7_payers": d7_payers,
        "predicted_late": predicted_late,
        "enriched": enriched,
        "oracle": oracle,
        "enriched_from_d7": enriched_from_d7,
        "enriched_from_late": enriched_from_late,
        "whale_threshold": whale_threshold,
        "total_rev": total_rev,
        "has_engagement": len(eng_cols) > 0,
    }, None


def list_available_datasets():
    datasets = {}
    for f in DATA_DIR.glob("cfm_pltv*.csv"):
        size_mb = f.stat().st_size / 1e6
        datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    return datasets


# ── page ───────────────────────────────────────────────────────────
st.title("🌱 Seed Optimization")
st.markdown(
    "Compare **seed list strategies** for UA lookalike expansion. "
    "Should we include predicted late payers (rev_d7=0 but high engagement) to improve seed quality?"
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
col_ds1, col_ds2, col_ds3 = st.columns([2, 2, 1])
with col_ds1:
    chosen_ds = st.selectbox("Dataset", ds_names, index=default_idx, key="seed_dataset",
                             help="Choose which dataset to analyze")
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.1f} MB")
with col_ds3:
    top_late_pct = st.number_input("Top % late payers", min_value=1, max_value=20, value=5,
                                    help="Top N% of D7=0 users by engagement score added to enriched seed")

df_raw, metrics, error = compute_seed_metrics(ds_info["path"], ds_info["mtime"], top_late_pct)
if error:
    st.error(f"❌ {error}")
    st.stop()

strategies = metrics["strategies"]
n_users = len(df_raw)
st.success(f"✅ Loaded **{n_users:,}** users — evaluating 4 seed strategies")

if not metrics["has_engagement"]:
    st.warning("⚠️ No engagement features found — using rev_d7 only for scoring. Results may be limited.")

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Seed_Optimization_Strategy.md", "📄 Full Seed Optimization Report")

# ── KPI Summary ────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Strategy Comparison")

d7_row = strategies[strategies["Strategy"] == "D7 Payers Only"].iloc[0]
enriched_row = strategies[strategies["Strategy"].str.startswith("D7 Payers + Top")].iloc[0]
oracle_row = strategies[strategies["Strategy"] == "D30 Payers (Oracle)"].iloc[0]

k1, k2, k3, k4 = st.columns(4)
with k1:
    size_gain = enriched_row["Seed Size"] - d7_row["Seed Size"]
    st.metric("Enriched Seed Size", f"{int(enriched_row['Seed Size']):,}",
              f"+{size_gain:,} vs D7-only")
with k2:
    whale_gain = enriched_row["Whale Capture"] - d7_row["Whale Capture"]
    st.metric("Whale Capture (Enriched)", f"{enriched_row['Whale Capture']:.1%}",
              f"+{whale_gain:.1%} vs D7-only")
with k3:
    rev_gap = oracle_row["Total Revenue"] - d7_row["Total Revenue"]
    st.metric("Revenue Gap to Oracle",
              format_currency(convert_vnd(rev_gap, cur["code"]), cur["code"]),
              "missed by D7-only approach")
with k4:
    st.metric("Oracle Whale Capture", f"{oracle_row['Whale Capture']:.1%}", "theoretical max")

# ── Chart 1: Strategy Comparison ──────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    fig_ltv = px.bar(
        strategies, x="Strategy",
        y=convert_vnd(strategies["Avg LTV30"], cur["code"]),
        title=f"Avg LTV30 per Seed User ({cur['symbol']})",
        color="Strategy",
        color_discrete_sequence=["#FF6600", "#2ecc71", "royalblue", "#9b59b6"],
        labels={"y": f"Avg LTV30 ({cur['symbol']})", "Strategy": ""},
    )
    fig_ltv.update_layout(height=420, showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_ltv, use_container_width=True)

with col2:
    fig_whale = px.bar(
        strategies, x="Strategy",
        y=strategies["Whale Capture"] * 100,
        title="Whale Capture Rate by Strategy (%)",
        color="Strategy",
        color_discrete_sequence=["#FF6600", "#2ecc71", "royalblue", "#9b59b6"],
        labels={"y": "Whale Capture (%)", "Strategy": ""},
    )
    fig_whale.update_layout(height=420, showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_whale, use_container_width=True)

# ── Chart 2: Size vs Quality ───────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    fig_tradeoff = px.scatter(
        strategies, x="Seed Size",
        y=convert_vnd(strategies["Avg LTV30"], cur["code"]),
        text="Strategy", size=[40] * len(strategies),
        color="Strategy",
        color_discrete_sequence=["#FF6600", "#2ecc71", "royalblue", "#9b59b6"],
        title="Seed Size vs Quality Tradeoff",
        labels={"x": "Seed Size (users)", "y": f"Avg LTV30 ({cur['symbol']})"},
    )
    fig_tradeoff.update_traces(textposition="top center")
    fig_tradeoff.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

with col4:
    # Revenue composition of enriched seed
    comp_labels = ["D7 Payers Revenue", f"Predicted Late Payer Revenue"]
    comp_values = [
        convert_vnd(metrics["enriched_from_d7"], cur["code"]),
        convert_vnd(metrics["enriched_from_late"], cur["code"]),
    ]
    fig_comp = go.Figure(go.Pie(
        labels=comp_labels, values=comp_values,
        marker_colors=["#FF6600", "#2ecc71"],
        hole=0.4,
    ))
    fig_comp.update_layout(
        title=f"Enriched Seed Revenue Composition ({cur['symbol']})",
        height=420,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ── Summary Table ──────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Strategy Summary Table")
tbl = strategies.copy()
tbl["Avg LTV30"] = tbl["Avg LTV30"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["Total Revenue"] = tbl["Total Revenue"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["Payer Rate"] = (tbl["Payer Rate"] * 100).round(1).astype(str) + "%"
tbl["Whale Capture"] = (tbl["Whale Capture"] * 100).round(1).astype(str) + "%"
tbl["Seed Size"] = tbl["Seed Size"].apply(lambda v: f"{int(v):,}")
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
size_pct = (enriched_row["Seed Size"] - d7_row["Seed Size"]) / d7_row["Seed Size"] * 100 if d7_row["Seed Size"] > 0 else 0
late_rev_pct = metrics["enriched_from_late"] / (metrics["enriched_from_d7"] + metrics["enriched_from_late"]) * 100 if (metrics["enriched_from_d7"] + metrics["enriched_from_late"]) > 0 else 0

st.markdown(f"- **Enriched seed** is **{size_pct:.0f}% larger** than D7-only, capturing **{enriched_row['Whale Capture']:.1%}** of whales vs {d7_row['Whale Capture']:.1%}")
st.markdown(f"- Predicted late payers contribute **{late_rev_pct:.1f}%** of enriched seed total revenue")
st.markdown(f"- **Revenue gap to oracle:** {format_currency(convert_vnd(oracle_row['Total Revenue'] - d7_row['Total Revenue'], cur['code']), cur['code'])} missed by D7-only approach")
st.markdown(f"- **Recommendation:** Implement enriched seeds with top {top_late_pct}% predicted late payers — larger seeds improve network learning without significantly diluting quality")
