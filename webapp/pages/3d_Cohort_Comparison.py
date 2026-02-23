"""
Page 3d — Cohort Comparison
Compare user cohorts by media source and OS to identify best acquisition channels.
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
    render_dataset_role_selector,
)

render_top_menu()
render_sidebar()


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing cohort metrics…")
def compute_cohort_metrics(csv_path: str, file_mtime: float, _df: pd.DataFrame = None):
    df = _df if _df is not None else pd.read_csv(csv_path, low_memory=False)
    required = {"ltv30", "media_source"}
    missing = required - set(df.columns)
    if missing:
        return None, None, f"Dataset missing columns: {missing}"

    df["is_payer_30"] = (df["ltv30"] > 0).astype(int)
    df["rev_d7"] = df.get("rev_d7", pd.Series(0.0, index=df.index)).astype(float)
    df["is_payer_d7"] = (df["rev_d7"] > 0).astype(int)
    df["is_late_payer"] = ((df["rev_d7"] == 0) & (df["ltv30"] > 0)).astype(int)

    agg = {
        "users": ("ltv30", "count"),
        "arpu_d30": ("ltv30", "mean"),
        "total_ltv30": ("ltv30", "sum"),
        "payer_rate_d30": ("is_payer_30", "mean"),
        "payer_rate_d7": ("is_payer_d7", "mean"),
        "late_payer_rate": ("is_late_payer", "mean"),
    }
    if "games_d7" in df.columns:
        agg["avg_games"] = ("games_d7", "mean")
    if "active_days_d7" in df.columns:
        agg["avg_active_days"] = ("active_days_d7", "mean")

    by_source = df.groupby("media_source").agg(**agg).reset_index()
    by_source = by_source.sort_values("arpu_d30", ascending=False)

    by_os = None
    if "first_os" in df.columns:
        by_os = df.groupby("first_os").agg(**agg).reset_index().sort_values("arpu_d30", ascending=False)

    return df, {"by_source": by_source, "by_os": by_os}, None


# ── page ───────────────────────────────────────────────────────────
st.title("👥 Cohort Comparison")
st.markdown(
    "Compare user cohorts by **media source** and **OS** to identify which acquisition channels "
    "deliver the highest LTV and where late payer detection adds the most value."
)

cur = get_currency_info()

# ── Dataset role selector ─────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Dataset")
_df_loaded = render_dataset_role_selector(
    page_key="cc",
    help_text="**Both** combines train + test for more media sources and wider coverage.",
)
st.markdown("---")

_cache_key = f"{len(_df_loaded)}_{list(_df_loaded.columns)[:5]}"
df_raw, metrics, error = compute_cohort_metrics(_cache_key, 0.0, _df=_df_loaded)
if error:
    st.error(f"❌ {error}")
    st.stop()

by_source = metrics["by_source"]
by_os = metrics["by_os"]
n_users = len(df_raw)
st.success(f"✅ Loaded **{n_users:,}** users across **{by_source['media_source'].nunique()}** media sources")

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Cohort_Comparison.md", "📄 Full Cohort Comparison Report")

# ── KPI Summary ────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Overall KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Users", f"{n_users:,}")
with k2:
    st.metric("Media Sources", f"{by_source['media_source'].nunique()}")
with k3:
    best_src = by_source.iloc[0]
    st.metric("Best ARPU Channel", best_src["media_source"],
              format_currency(best_src["arpu_d30"], cur["code"]))
with k4:
    overall_payer = df_raw["is_payer_30"].mean()
    st.metric("Overall D30 Payer Rate", f"{overall_payer:.2%}")
with k5:
    overall_late = df_raw["is_late_payer"].mean()
    st.metric("Overall Late Payer Rate", f"{overall_late:.2%}")

# ── Chart 1: ARPU by Source ────────────────────────────────────────
st.markdown("---")
st.header("📈 ARPU & Payer Rates by Media Source")
col1, col2 = st.columns(2)

with col1:
    fig_arpu = px.bar(
        by_source, x="media_source",
        y=convert_vnd(by_source["arpu_d30"], cur["code"]),
        title=f"ARPU D30 by Media Source ({cur['symbol']})",
        color="arpu_d30", color_continuous_scale="Blues",
        labels={"y": f"ARPU ({cur['symbol']})", "media_source": ""},
    )
    fig_arpu.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                           xaxis_tickangle=-30)
    st.plotly_chart(fig_arpu, use_container_width=True)

with col2:
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Bar(
        x=by_source["media_source"], y=by_source["payer_rate_d30"] * 100,
        name="D30 Payer %", marker_color="royalblue",
    ))
    fig_rates.add_trace(go.Bar(
        x=by_source["media_source"], y=by_source["payer_rate_d7"] * 100,
        name="D7 Payer %", marker_color="#e74c3c",
    ))
    fig_rates.add_trace(go.Bar(
        x=by_source["media_source"], y=by_source["late_payer_rate"] * 100,
        name="Late Payer %", marker_color="#2ecc71",
    ))
    fig_rates.update_layout(
        title="Payer Rates by Media Source (%)", barmode="group",
        height=420, xaxis_tickangle=-30,
        legend=dict(orientation="h", y=-0.3),
        yaxis_title="Rate (%)",
    )
    st.plotly_chart(fig_rates, use_container_width=True)

# ── Chart 2: Size vs Quality ───────────────────────────────────────
st.markdown("---")
st.header("🎯 Size vs Quality Tradeoff")
col3, col4 = st.columns(2)

with col3:
    fig_scatter = px.scatter(
        by_source, x="users", y=convert_vnd(by_source["arpu_d30"], cur["code"]),
        size="total_ltv30", color="media_source", text="media_source",
        title="Users vs ARPU (bubble = total revenue)",
        labels={"x": "Users", "y": f"ARPU ({cur['symbol']})"},
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    fig_late = px.bar(
        by_source.sort_values("late_payer_rate", ascending=False),
        x="media_source", y=by_source.sort_values("late_payer_rate", ascending=False)["late_payer_rate"] * 100,
        title="Late Payer Rate by Source (ML Opportunity)",
        color="late_payer_rate", color_continuous_scale="Greens",
        labels={"y": "Late Payer Rate (%)", "media_source": ""},
    )
    fig_late.update_layout(height=420, coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_late, use_container_width=True)

# ── Chart 3: OS Comparison ─────────────────────────────────────────
if by_os is not None and len(by_os) > 1:
    st.markdown("---")
    st.header("📱 iOS vs Android Comparison")
    col5, col6 = st.columns(2)
    with col5:
        fig_os_arpu = px.bar(
            by_os, x="first_os", y=convert_vnd(by_os["arpu_d30"], cur["code"]),
            title=f"ARPU by OS ({cur['symbol']})",
            color="first_os", color_discrete_sequence=["#FF6600", "royalblue", "#2ecc71"],
            labels={"y": f"ARPU ({cur['symbol']})", "first_os": ""},
        )
        fig_os_arpu.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_os_arpu, use_container_width=True)
    with col6:
        fig_os_rates = go.Figure()
        fig_os_rates.add_trace(go.Bar(x=by_os["first_os"], y=by_os["payer_rate_d30"] * 100,
                                      name="D30 Payer %", marker_color="royalblue"))
        fig_os_rates.add_trace(go.Bar(x=by_os["first_os"], y=by_os["late_payer_rate"] * 100,
                                      name="Late Payer %", marker_color="#2ecc71"))
        fig_os_rates.update_layout(title="Payer Rates by OS (%)", barmode="group",
                                   height=380, yaxis_title="Rate (%)",
                                   legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_os_rates, use_container_width=True)

# ── LTV30 Distribution ─────────────────────────────────────────────
st.markdown("---")
st.header("📦 LTV30 Distribution by Source")
payers_only = df_raw[df_raw["ltv30"] > 0].copy()
if len(payers_only) > 0:
    fig_box = px.box(
        payers_only, x="media_source", y=convert_vnd(payers_only["ltv30"], cur["code"]),
        title=f"LTV30 Distribution — Payers Only ({cur['symbol']})",
        color="media_source",
        labels={"y": f"LTV30 ({cur['symbol']})", "media_source": ""},
    )
    fig_box.update_layout(height=450, showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Summary Table ──────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Media Source Summary Table")
tbl = by_source.copy()
tbl["arpu_d30"] = tbl["arpu_d30"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["payer_rate_d30"] = (tbl["payer_rate_d30"] * 100).round(2).astype(str) + "%"
tbl["payer_rate_d7"] = (tbl["payer_rate_d7"] * 100).round(2).astype(str) + "%"
tbl["late_payer_rate"] = (tbl["late_payer_rate"] * 100).round(2).astype(str) + "%"
tbl["users"] = tbl["users"].apply(lambda v: f"{v:,}")
cols = ["media_source", "users", "arpu_d30", "payer_rate_d30", "payer_rate_d7", "late_payer_rate"]
if "avg_games" in tbl.columns:
    tbl["avg_games"] = tbl["avg_games"].round(1)
    cols.append("avg_games")
if "avg_active_days" in tbl.columns:
    tbl["avg_active_days"] = tbl["avg_active_days"].round(2)
    cols.append("avg_active_days")
tbl = tbl[cols]
tbl.columns = ["Media Source", "Users", f"ARPU ({cur['symbol']})", "D30 Payer %", "D7 Payer %", "Late Payer %"] + \
              (["Avg Games D7"] if "avg_games" in by_source.columns else []) + \
              (["Avg Active Days"] if "avg_active_days" in by_source.columns else [])
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
best = by_source.iloc[0]
worst = by_source.iloc[-1]
best_late = by_source.loc[by_source["late_payer_rate"].idxmax()]
arpu_range = by_source["arpu_d30"].max() / by_source["arpu_d30"].min() if by_source["arpu_d30"].min() > 0 else 0

st.markdown(f"- **Best ARPU channel:** {best['media_source']} — {format_currency(convert_vnd(best['arpu_d30'], cur['code']), cur['code'])} ARPU, {best['payer_rate_d30']:.2%} D30 payer rate")
st.markdown(f"- **Lowest ARPU channel:** {worst['media_source']} — {format_currency(convert_vnd(worst['arpu_d30'], cur['code']), cur['code'])} ARPU")
if arpu_range > 0:
    st.markdown(f"- **ARPU spread:** {arpu_range:.1f}x between best and worst channel — significant budget reallocation opportunity")
st.markdown(f"- **Highest late payer opportunity:** {best_late['media_source']} at {best_late['late_payer_rate']:.2%} late payer rate — benefits most from ML detection")
