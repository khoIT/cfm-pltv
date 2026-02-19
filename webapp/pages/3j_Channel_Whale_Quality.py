"""
Page 3j — Channel × Whale Quality
Evaluate UA channels by whale rate, ARPU, and engagement quality — not just volume.
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
    format_currency, DATA_DIR, REPORTS_DIR, list_analysis_datasets,
)

render_top_menu()
render_sidebar()


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Analysing channel whale quality…")
def compute_channel_metrics(csv_path: str, file_mtime: float, min_users: int = 50):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns or "media_source" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' and 'media_source' columns."

    df = df.copy()
    df["ltv30"] = pd.to_numeric(df["ltv30"], errors="coerce").fillna(0)
    df["media_source"] = df["media_source"].fillna("unknown").astype(str)

    p99 = df["ltv30"].quantile(0.99)
    p95 = df["ltv30"].quantile(0.95)
    total_rev = df["ltv30"].sum()

    df["is_payer"] = (df["ltv30"] > 0).astype(int)
    df["is_whale"] = (df["ltv30"] >= p95).astype(int)
    df["is_mega_whale"] = (df["ltv30"] >= p99).astype(int)

    eng_cols = [c for c in ["games_d7", "active_days_d7", "win_rate_d7",
                             "kd_d7", "max_level_seen_d7"] if c in df.columns]

    agg_dict = {
        "users": ("ltv30", "count"),
        "arpu_d30": ("ltv30", "mean"),
        "total_ltv30": ("ltv30", "sum"),
        "payer_rate": ("is_payer", "mean"),
        "whale_rate": ("is_whale", "mean"),
        "mega_whale_rate": ("is_mega_whale", "mean"),
    }
    for c in eng_cols:
        agg_dict[f"avg_{c}"] = (c, "mean")

    by_channel = df.groupby("media_source").agg(**agg_dict).reset_index()
    by_channel = by_channel[by_channel["users"] >= min_users].copy()
    by_channel["rev_share_%"] = (by_channel["total_ltv30"] / total_rev * 100).round(1)
    by_channel["whale_rate_%"] = (by_channel["whale_rate"] * 100).round(2)
    by_channel["mega_whale_rate_%"] = (by_channel["mega_whale_rate"] * 100).round(3)
    by_channel["payer_rate_%"] = (by_channel["payer_rate"] * 100).round(1)
    by_channel = by_channel.sort_values("whale_rate_%", ascending=False).reset_index(drop=True)

    # Organic benchmark
    organic = by_channel[by_channel["media_source"] == "organic"]
    organic_whale_rate = organic["whale_rate_%"].values[0] if len(organic) > 0 else None
    if organic_whale_rate and organic_whale_rate > 0:
        by_channel["whale_rate_vs_organic"] = (
            by_channel["whale_rate_%"] / organic_whale_rate * 100).round(1)
    else:
        by_channel["whale_rate_vs_organic"] = None

    # OS breakdown if available
    by_os = None
    if "first_os" in df.columns:
        by_os = df.groupby("first_os").agg(**agg_dict).reset_index()
        by_os["whale_rate_%"] = (by_os["whale_rate"] * 100).round(2)
        by_os["payer_rate_%"] = (by_os["payer_rate"] * 100).round(1)
        by_os = by_os.sort_values("whale_rate_%", ascending=False)

    return df, {
        "by_channel": by_channel,
        "by_os": by_os,
        "eng_cols": eng_cols,
        "p95": p95, "p99": p99,
        "organic_whale_rate": organic_whale_rate,
    }, None


# ── page ───────────────────────────────────────────────────────────
st.title("📡 Channel × Whale Quality")
st.markdown(
    "Evaluate UA channels by **whale rate** — the fraction of acquired users who become "
    "top revenue contributors. Goes beyond CPI and ROAS to measure true acquisition quality."
)

cur = get_currency_info()

# ── Dataset selector ───────────────────────────────────────────────
st.header("📂 Select Dataset")
datasets, default_idx = list_analysis_datasets()
if not datasets:
    st.error("No datasets found in data/ directory.")
    st.stop()

ds_names = list(datasets.keys())
col_ds1, col_ds2, col_ds3 = st.columns([2, 2, 1])
with col_ds1:
    chosen_ds = st.selectbox("Dataset", ds_names, index=default_idx, key="channel_dataset")
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.0f} MB")
with col_ds3:
    min_users = st.number_input("Min users/channel", min_value=10, max_value=500,
                                 value=50, step=10, key="channel_min_users")

with st.spinner("Analysing channel whale quality…"):
    df_raw, metrics, error = compute_channel_metrics(
        ds_info["path"], ds_info["mtime"], min_users=int(min_users))

if error:
    st.error(f"❌ {error}")
    st.stop()

by_channel = metrics["by_channel"]
by_os = metrics["by_os"]
eng_cols = metrics["eng_cols"]
organic_whale_rate = metrics["organic_whale_rate"]

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Channel_Whale_Quality.md", "📄 Full Channel × Whale Quality Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Channel Overview")

best_whale_ch = by_channel.iloc[0] if len(by_channel) > 0 else None
top_arpu_ch = by_channel.sort_values("arpu_d30", ascending=False).iloc[0] if len(by_channel) > 0 else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Channels Analysed", f"{len(by_channel)}", f"≥{min_users} users each")
with k2:
    if best_whale_ch is not None:
        st.metric("Best Whale Rate Channel", best_whale_ch["media_source"],
                  f"{best_whale_ch['whale_rate_%']:.2f}% whale rate")
with k3:
    if organic_whale_rate:
        st.metric("Organic Whale Rate (benchmark)", f"{organic_whale_rate:.2f}%")
with k4:
    if top_arpu_ch is not None:
        st.metric("Highest ARPU Channel", top_arpu_ch["media_source"],
                  format_currency(convert_vnd(top_arpu_ch["arpu_d30"], cur["code"]), cur["code"]))

# ── Whale Rate by Channel ───────────────────────────────────────────
st.markdown("---")
st.header("🐋 Whale Rate by Channel")
col1, col2 = st.columns(2)

with col1:
    channel_colors = px.colors.qualitative.Set2[:len(by_channel)]
    fig_whale = go.Figure()
    fig_whale.add_trace(go.Bar(
        x=by_channel["media_source"],
        y=by_channel["whale_rate_%"],
        marker_color=channel_colors,
        text=by_channel["whale_rate_%"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
        name="Whale Rate",
    ))
    if organic_whale_rate:
        fig_whale.add_hline(y=organic_whale_rate, line_dash="dash", line_color="green",
                            annotation_text=f"Organic: {organic_whale_rate:.2f}%")
    fig_whale.update_layout(
        title="Whale Rate (top 5% LTV30) by Channel",
        yaxis_title="Whale Rate (%)", height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_whale, use_container_width=True)

with col2:
    fig_arpu = go.Figure()
    arpu_sorted = by_channel.sort_values("arpu_d30", ascending=False)
    fig_arpu.add_trace(go.Bar(
        x=arpu_sorted["media_source"],
        y=arpu_sorted["arpu_d30"].apply(lambda v: convert_vnd(v, cur["code"])),
        marker_color=px.colors.qualitative.Set2[:len(arpu_sorted)],
        text=arpu_sorted["arpu_d30"].apply(
            lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"])),
        textposition="outside",
    ))
    fig_arpu.update_layout(
        title=f"ARPU D30 by Channel ({cur['symbol']})",
        yaxis_title=f"ARPU D30 ({cur['symbol']})", height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_arpu, use_container_width=True)

# ── Whale Rate vs ARPU Scatter ──────────────────────────────────────
st.markdown("---")
st.header("📊 Whale Rate vs ARPU — Channel Positioning")

fig_scatter = px.scatter(
    by_channel,
    x="whale_rate_%", y=by_channel["arpu_d30"].apply(lambda v: convert_vnd(v, cur["code"])),
    size="users", color="media_source",
    text="media_source",
    labels={"x": "Whale Rate (%)", "y": f"ARPU D30 ({cur['symbol']})",
            "media_source": "Channel"},
    title="Channel Quality Matrix: Whale Rate vs ARPU (bubble = user volume)",
    height=450,
)
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(showlegend=False)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Payer Rate vs Whale Rate ────────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    fig_payer = go.Figure()
    fig_payer.add_trace(go.Bar(
        x=by_channel["media_source"], y=by_channel["payer_rate_%"],
        marker_color=channel_colors, name="Payer Rate",
        text=by_channel["payer_rate_%"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig_payer.update_layout(
        title="Payer Rate (D30) by Channel",
        yaxis_title="Payer Rate (%)", height=380, showlegend=False,
    )
    st.plotly_chart(fig_payer, use_container_width=True)

with col4:
    fig_rev = go.Figure()
    rev_sorted = by_channel.sort_values("rev_share_%", ascending=False)
    fig_rev.add_trace(go.Bar(
        x=rev_sorted["media_source"], y=rev_sorted["rev_share_%"],
        marker_color=px.colors.qualitative.Set2[:len(rev_sorted)],
        text=rev_sorted["rev_share_%"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig_rev.update_layout(
        title="Revenue Share (% of total LTV30) by Channel",
        yaxis_title="Revenue Share (%)", height=380, showlegend=False,
    )
    st.plotly_chart(fig_rev, use_container_width=True)

# ── Engagement Quality ──────────────────────────────────────────────
if eng_cols:
    st.markdown("---")
    st.header("🎮 Engagement Quality by Channel")
    eng_disp_cols = [f"avg_{c}" for c in eng_cols if f"avg_{c}" in by_channel.columns]
    if eng_disp_cols:
        eng_tbl = by_channel[["media_source", "users"] + eng_disp_cols].copy()
        eng_tbl.columns = (["Channel", "Users"] +
                           [c.replace("avg_", "").replace("_d7", "") for c in eng_disp_cols])
        st.dataframe(eng_tbl, use_container_width=True, hide_index=True)

# ── OS Breakdown ────────────────────────────────────────────────────
if by_os is not None and len(by_os) > 0:
    st.markdown("---")
    st.header("📱 OS Breakdown")
    col5, col6 = st.columns(2)
    with col5:
        fig_os_whale = go.Figure()
        fig_os_whale.add_trace(go.Bar(
            x=by_os["first_os"], y=by_os["whale_rate_%"],
            marker_color=["#3498db", "#e74c3c", "#2ecc71"][:len(by_os)],
            text=by_os["whale_rate_%"].apply(lambda v: f"{v:.2f}%"),
            textposition="outside",
        ))
        fig_os_whale.update_layout(title="Whale Rate by OS", yaxis_title="Whale Rate (%)",
                                    height=350, showlegend=False)
        st.plotly_chart(fig_os_whale, use_container_width=True)
    with col6:
        fig_os_arpu = go.Figure()
        fig_os_arpu.add_trace(go.Bar(
            x=by_os["first_os"],
            y=by_os["arpu_d30"].apply(lambda v: convert_vnd(v, cur["code"])),
            marker_color=["#3498db", "#e74c3c", "#2ecc71"][:len(by_os)],
            text=by_os["arpu_d30"].apply(
                lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"])),
            textposition="outside",
        ))
        fig_os_arpu.update_layout(
            title=f"ARPU D30 by OS ({cur['symbol']})",
            yaxis_title=f"ARPU ({cur['symbol']})", height=350, showlegend=False)
        st.plotly_chart(fig_os_arpu, use_container_width=True)

# ── Summary Table ───────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Channel Summary Table")
tbl = by_channel[["media_source", "users", "payer_rate_%", "whale_rate_%",
                   "mega_whale_rate_%", "arpu_d30", "rev_share_%"]].copy()
tbl["arpu_d30"] = tbl["arpu_d30"].apply(
    lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
if "whale_rate_vs_organic" in by_channel.columns:
    tbl["vs_organic"] = by_channel["whale_rate_vs_organic"].apply(
        lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
    tbl.columns = ["Channel", "Users", "Payer Rate %", "Whale Rate %",
                   "Mega-Whale Rate %", f"ARPU ({cur['symbol']})", "Rev Share %", "vs Organic"]
else:
    tbl.columns = ["Channel", "Users", "Payer Rate %", "Whale Rate %",
                   "Mega-Whale Rate %", f"ARPU ({cur['symbol']})", "Rev Share %"]
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
if best_whale_ch is not None:
    st.markdown(f"- **Best whale rate channel:** {best_whale_ch['media_source']} "
                f"({best_whale_ch['whale_rate_%']:.2f}% whale rate)")
if organic_whale_rate:
    st.markdown(f"- **Organic benchmark:** {organic_whale_rate:.2f}% whale rate — "
                "channels above this deliver organic-quality users at paid scale")
st.markdown("- **Whale rate ≠ payer rate** — a channel can have high payer rate but low whale rate")
st.markdown("- **Revenue share** is the ultimate channel quality metric — not just volume")
st.markdown("### 🎯 Recommended Actions")
st.markdown("- Report **whale rate** alongside CPI and ROAS in all campaign dashboards")
st.markdown("- Shift budget toward channels with whale rate ≥ 80% of organic benchmark")
st.markdown("- Build channel-specific lookalike seeds from each channel's whale users")
