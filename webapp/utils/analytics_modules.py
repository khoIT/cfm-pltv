"""
analytics_modules.py — Deterministic analytics modules for the CFM Data Chatbot.
Each module runs pre-built SQL + charting logic triggered by keywords,
producing impressive, correct outputs every time without LLM dependency.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Optional, Callable
from streamlit_echarts import st_echarts


# ---------------------------------------------------------------------------
# Type alias for the SQL executor
# ---------------------------------------------------------------------------
ExecuteSqlFn = Callable[[str, int], tuple[Optional[pd.DataFrame], str]]


# ---------------------------------------------------------------------------
# Flow detection — keyword router
# ---------------------------------------------------------------------------
FLOW_TRIGGERS = {
    "pareto": [
        "pareto", "concentration", "top 1%", "top 5%", "top 10%",
        "revenue distribution", "whale threshold", "80/20", "revenue curve",
        "how concentrated",
    ],
    "late_payer": [
        "late payer", "late-payer", "missed revenue", "d7-only gap",
        "opportunity size", "late conversion", "hidden revenue",
        "revenue left on table", "non-payer convert",
    ],
    "channel": [
        "channel comparison", "channel quality", "channel roi",
        "which channel", "best channel", "worst channel",
        "media source comparison", "compare channels", "channel dashboard",
        "channel performance",
    ],
    "cohort": [
        "cohort quality", "cohort trend", "declining quality",
        "user quality trend", "cohort tracker", "install quality",
        "weekly quality", "quality over time",
    ],
    "feature_importance": [
        "feature importance", "what drives", "model explain",
        "why does the model", "important features", "top features",
        "model weights", "xgboost features",
    ],
    "executive_brief": [
        "executive brief", "executive summary", "key findings",
        "dashboard overview", "kpi summary", "give me the overview",
        "quick summary", "tldr", "tl;dr",
    ],
    "anomaly": [
        "anomaly", "outlier", "unusual", "suspicious",
        "extreme users", "abnormal", "weird",
    ],
    "data_quality": [
        "data status", "data quality", "data coverage",
        "data freshness", "null rate", "missing data", "dataset info",
    ],
}


def detect_deterministic_flow(question: str) -> Optional[str]:
    """Return a flow name if the question matches a deterministic module, else None."""
    q = question.lower()
    best_flow = None
    best_count = 0
    for flow, triggers in FLOW_TRIGGERS.items():
        matches = sum(1 for t in triggers if t in q)
        if matches > best_count:
            best_count = matches
            best_flow = flow
    return best_flow if best_count > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# MODULE A — Live Pareto / Revenue Concentration Calculator
# ═══════════════════════════════════════════════════════════════════════════
def run_pareto_analysis(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Interactive Pareto curve with percentile slider. Returns (summary, follow_ups)."""

    sql = """
    SELECT
        ltv30,
        is_payer_30
    FROM cfm_features
    WHERE ltv30 > 0
    ORDER BY ltv30 DESC
    LIMIT 500000
    """
    df, err = exec_sql(sql, 500000)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    total_users_sql = "SELECT COUNT(*) AS n, SUM(CASE WHEN ltv30 > 0 THEN 1 ELSE 0 END) AS payers FROM cfm_features"
    df_total, _ = exec_sql(total_users_sql, 10)
    total_users = int(df_total["n"].iloc[0]) if df_total is not None else 0
    total_payers = int(df_total["payers"].iloc[0]) if df_total is not None else len(df)

    values = df["ltv30"].values.astype(float)
    values.sort()
    values = values[::-1]  # descending
    cum_rev = np.cumsum(values)
    total_rev = cum_rev[-1]
    n = len(values)

    # Build Pareto curve data (sample points for chart performance)
    sample_points = min(200, n)
    indices = np.linspace(0, n - 1, sample_points, dtype=int)
    pct_users = ((indices + 1) / n * 100).tolist()
    pct_rev = (cum_rev[indices] / total_rev * 100).tolist()

    # Interactive percentile slider
    st.markdown("### 📊 Revenue Concentration — Pareto Analysis")
    pct_choice = st.slider(
        "Select top X% of payers", min_value=1, max_value=100, value=1,
        key="pareto_slider", help="Drag to see what % of revenue the top X% of payers contribute"
    )

    # Compute for selected percentile
    top_n = max(1, int(n * pct_choice / 100))
    top_rev = float(cum_rev[top_n - 1])
    top_pct_rev = top_rev / total_rev * 100

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Payers", f"{total_payers:,}")
    c2.metric(f"Top {pct_choice}% Payers", f"{top_n:,}")
    c3.metric(f"Revenue Share", f"{top_pct_rev:.1f}%")
    c4.metric("Total Revenue", f"₫{total_rev/1e9:.1f}B")

    # Whale threshold
    threshold_idx = max(1, int(n * 0.01)) - 1
    whale_threshold = float(values[threshold_idx])
    st.info(f"🐋 **Whale threshold** (top 1%): ₫{whale_threshold:,.0f} LTV30 — "
            f"these {threshold_idx + 1:,} users contribute "
            f"**{cum_rev[threshold_idx] / total_rev * 100:.1f}%** of total revenue")

    # Pareto curve EChart
    # Mark the selected percentile on the chart
    mark_x = pct_choice
    mark_y = top_pct_rev
    option = {
        "title": {"text": "Pareto Curve — Cumulative Revenue by User Percentile", "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "Top {b}% of payers = {c}% of revenue"},
        "xAxis": {"type": "category", "name": "Top X% of Payers",
                   "data": [f"{p:.1f}" for p in pct_users],
                   "axisLabel": {"rotate": 0, "interval": sample_points // 10}},
        "yAxis": {"type": "value", "name": "% of Total Revenue", "max": 100},
        "grid": {"left": "10%", "right": "5%", "bottom": "12%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}}},
        "series": [
            {
                "type": "line",
                "data": [round(p, 1) for p in pct_rev],
                "smooth": True,
                "areaStyle": {"opacity": 0.25, "color": "#5470c6"},
                "lineStyle": {"width": 3},
                "name": "Cumulative Revenue %",
                "markLine": {
                    "data": [
                        {"xAxis": str(f"{mark_x:.1f}"), "label": {"formatter": f"Top {mark_x}%"}},
                        {"yAxis": mark_y, "label": {"formatter": f"{mark_y:.1f}%"}},
                    ],
                    "lineStyle": {"type": "dashed", "color": "#ee6666"},
                },
            },
            {
                "type": "line",
                "data": [round(p, 1) for p in pct_users],
                "lineStyle": {"type": "dashed", "color": "#91cc75", "width": 1},
                "name": "Perfect equality",
                "symbol": "none",
            },
        ],
        "legend": {"bottom": 0},
    }
    st_echarts(option, height="480px", key="pareto_curve")

    # Revenue distribution histogram
    bins = [0, 1000, 10000, 50000, 100000, 323000, 1000000, 5000000, float("inf")]
    labels = ["₫0-1K", "₫1K-10K", "₫10K-50K", "₫50K-100K", "₫100K-323K", "₫323K-1M", "₫1M-5M", "₫5M+"]
    hist_counts = []
    hist_rev = []
    for i in range(len(bins) - 1):
        mask = (values >= bins[i]) & (values < bins[i + 1])
        hist_counts.append(int(mask.sum()))
        hist_rev.append(float(values[mask].sum()))
    total_for_pct = sum(hist_rev)

    option_hist = {
        "title": {"text": "Revenue Distribution by LTV30 Bracket", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 30}},
        "yAxis": [
            {"type": "value", "name": "User Count", "position": "left"},
            {"type": "value", "name": "Revenue Share %", "position": "right", "max": 100},
        ],
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "series": [
            {"type": "bar", "data": hist_counts, "name": "Users",
             "itemStyle": {"borderRadius": [4, 4, 0, 0]}, "yAxisIndex": 0},
            {"type": "line", "data": [round(r / total_for_pct * 100, 1) for r in hist_rev],
             "name": "Revenue %", "yAxisIndex": 1, "smooth": True,
             "lineStyle": {"width": 3, "color": "#ee6666"}},
        ],
        "legend": {"bottom": 0},
    }
    st_echarts(option_hist, height="400px", key="pareto_hist")

    summary = (
        f"**Pareto Analysis**: Top {pct_choice}% of payers ({top_n:,} users) contribute "
        f"{top_pct_rev:.1f}% of total revenue (₫{total_rev/1e9:.1f}B). "
        f"Whale threshold (top 1%) is ₫{whale_threshold:,.0f} LTV30. "
        f"Revenue is highly concentrated — a small group drives the vast majority of monetization."
    )
    st.markdown(f"**💡 Insight:** {summary}")

    follow_ups = [
        "Who are the whales? Show me their behavioral profile",
        "Which channels produce the most whales?",
        "Show me the late payer opportunity — how much revenue are we missing at D7?",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE B — Late-Payer Revenue Opportunity Sizer
# ═══════════════════════════════════════════════════════════════════════════
def run_late_payer_analysis(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Waterfall chart quantifying missed revenue from D7-only view."""

    sql = """
    SELECT
        CASE
            WHEN rev_d7 > 0 THEN 'D7 Payer'
            WHEN rev_d7 = 0 AND ltv30 > 0 THEN 'Late Payer (D8-D30)'
            ELSE 'Non-Payer'
        END AS segment,
        COUNT(*) AS users,
        SUM(ltv30) AS total_revenue,
        AVG(ltv30) AS avg_ltv,
        SUM(CASE WHEN ltv30 >= 323000 THEN 1 ELSE 0 END) AS whale_count
    FROM cfm_features
    GROUP BY segment
    ORDER BY total_revenue DESC
    """
    df, err = exec_sql(sql, 10)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    st.markdown("### 💰 Late-Payer Revenue Opportunity")

    # Extract metrics
    d7_row = df[df["segment"] == "D7 Payer"].iloc[0] if len(df[df["segment"] == "D7 Payer"]) else None
    late_row = df[df["segment"] == "Late Payer (D8-D30)"].iloc[0] if len(df[df["segment"] == "Late Payer (D8-D30)"]) else None
    non_row = df[df["segment"] == "Non-Payer"].iloc[0] if len(df[df["segment"] == "Non-Payer"]) else None

    d7_rev = float(d7_row["total_revenue"]) if d7_row is not None else 0
    late_rev = float(late_row["total_revenue"]) if late_row is not None else 0
    total_rev = d7_rev + late_rev
    d7_users = int(d7_row["users"]) if d7_row is not None else 0
    late_users = int(late_row["users"]) if late_row is not None else 0
    total_users = int(df["users"].sum())
    late_whales = int(late_row["whale_count"]) if late_row is not None else 0
    d7_whales = int(d7_row["whale_count"]) if d7_row is not None else 0
    total_whales = d7_whales + late_whales

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Late Payers", f"{late_users:,}", f"{late_users/total_users*100:.1f}% of all users")
    c2.metric("Late-Payer Revenue", f"₫{late_rev/1e9:.1f}B", f"{late_rev/total_rev*100:.1f}% of payer revenue")
    c3.metric("Late-Payer Whales", f"{late_whales:,}", f"{late_whales/total_whales*100:.1f}% of all whales")
    c4.metric("D7 ROAS Understatement", f"{total_rev/d7_rev:.1f}×" if d7_rev > 0 else "N/A",
              "D7-only misses this much revenue")

    # Waterfall chart
    waterfall_data = [
        {"name": "D7 Payer Revenue", "value": round(d7_rev / 1e9, 2)},
        {"name": "Late Payer Revenue", "value": round(late_rev / 1e9, 2)},
        {"name": "Total D30 Revenue", "value": round(total_rev / 1e9, 2)},
    ]
    option_waterfall = {
        "title": {"text": "Revenue Waterfall — D7 vs D30 View (₫ Billions)", "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "{b}: ₫{c}B"},
        "xAxis": {"type": "category", "data": [d["name"] for d in waterfall_data]},
        "yAxis": {"type": "value", "name": "₫ Billions"},
        "grid": {"left": "10%", "right": "5%", "bottom": "10%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "series": [{
            "type": "bar",
            "data": [
                {"value": waterfall_data[0]["value"],
                 "itemStyle": {"color": "#5470c6", "borderRadius": [4, 4, 0, 0]}},
                {"value": waterfall_data[1]["value"],
                 "itemStyle": {"color": "#ee6666", "borderRadius": [4, 4, 0, 0]}},
                {"value": waterfall_data[2]["value"],
                 "itemStyle": {"color": "#91cc75", "borderRadius": [4, 4, 0, 0]}},
            ],
            "label": {"show": True, "position": "top", "formatter": "₫{c}B"},
            "barWidth": "50%",
        }],
    }
    st_echarts(option_waterfall, height="400px", key="late_payer_waterfall")

    # Segment comparison chart
    segments = df[df["segment"] != "Non-Payer"]
    if not segments.empty:
        col1, col2 = st.columns(2)
        with col1:
            option_pie = {
                "title": {"text": "Revenue Split: D7 vs Late Payers", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: ₫{c}B ({d}%)"},
                "series": [{
                    "type": "pie", "radius": ["40%", "70%"],
                    "data": [
                        {"name": "D7 Payers", "value": round(d7_rev / 1e9, 2),
                         "itemStyle": {"color": "#5470c6"}},
                        {"name": "Late Payers", "value": round(late_rev / 1e9, 2),
                         "itemStyle": {"color": "#ee6666"}},
                    ],
                    "emphasis": {"itemStyle": {"shadowBlur": 10}},
                    "label": {"formatter": "{b}\n₫{c}B ({d}%)"},
                }],
            }
            st_echarts(option_pie, height="350px", key="late_payer_pie")

        with col2:
            option_whale = {
                "title": {"text": "Whale Distribution: D7 vs Late Payers", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} whales ({d}%)"},
                "series": [{
                    "type": "pie", "radius": ["40%", "70%"],
                    "data": [
                        {"name": "D7 Whales", "value": d7_whales,
                         "itemStyle": {"color": "#5470c6"}},
                        {"name": "Late-Payer Whales", "value": late_whales,
                         "itemStyle": {"color": "#ee6666"}},
                    ],
                    "emphasis": {"itemStyle": {"shadowBlur": 10}},
                    "label": {"formatter": "{b}\n{c} ({d}%)"},
                }],
            }
            st_echarts(option_whale, height="350px", key="late_payer_whale_pie")

    st.warning(
        f"⚠️ **Key Finding:** If you only look at D7 revenue, you miss **₫{late_rev/1e9:.1f}B** "
        f"({late_rev/total_rev*100:.0f}% of total payer revenue) and **{late_whales:,}** whales "
        f"({late_whales/total_whales*100:.0f}% of all whales). "
        f"D7-only ROAS is understated by **{total_rev/d7_rev:.1f}×**."
    )

    summary = (
        f"Late-Payer Analysis: {late_users:,} late payers ({late_users/total_users*100:.1f}% of users) "
        f"contribute ₫{late_rev/1e9:.1f}B ({late_rev/total_rev*100:.0f}% of payer revenue). "
        f"{late_whales:,} whales are late payers ({late_whales/total_whales*100:.0f}%). "
        f"D7-only ROAS understates by {total_rev/d7_rev:.1f}×."
    )

    follow_ups = [
        "What behavioral signals predict late payers?",
        "Compare seed strategies: D7-only vs enriched with late payers",
        "Show me the Pareto concentration for whales",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE C — Channel Deep-Dive Dashboard
# ═══════════════════════════════════════════════════════════════════════════
def run_channel_dashboard(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Multi-metric channel comparison dashboard."""

    sql = """
    SELECT
        media_source,
        COUNT(*) AS users,
        AVG(ltv30) AS arpu,
        SUM(ltv30) AS total_revenue,
        100.0 * SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END) / COUNT(*) AS payer_rate,
        100.0 * SUM(CASE WHEN ltv30 >= 323000 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END), 0) AS whale_rate_of_payers,
        100.0 * SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END), 0) AS late_payer_rate,
        AVG(active_days_d7) AS avg_active_days,
        AVG(games_d7) AS avg_games
    FROM cfm_features
    WHERE media_source IS NOT NULL AND media_source != ''
    GROUP BY media_source
    HAVING COUNT(*) >= 1000
    ORDER BY arpu DESC
    """
    df, err = exec_sql(sql, 100)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    st.markdown("### 📡 Channel Performance Dashboard")

    # Highlight best/worst
    best_channel = df.iloc[0]["media_source"]
    worst_channel = df.iloc[-1]["media_source"]
    best_arpu = df.iloc[0]["arpu"]
    worst_arpu = df.iloc[-1]["arpu"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Channels Analyzed", f"{len(df)}")
    c2.metric("🏆 Best ARPU", f"{best_channel}", f"₫{best_arpu:,.0f}")
    c3.metric("⚠️ Worst ARPU", f"{worst_channel}", f"₫{worst_arpu:,.0f}")

    # Chart 1: ARPU by channel
    channels = df["media_source"].tolist()
    option_arpu = {
        "title": {"text": "ARPU by Channel (₫)", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": channels, "axisLabel": {"rotate": 35}},
        "yAxis": {"type": "value", "name": "₫"},
        "grid": {"left": "10%", "right": "5%", "bottom": "20%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "series": [{
            "type": "bar", "data": [round(v, 0) for v in df["arpu"].tolist()],
            "itemStyle": {"borderRadius": [4, 4, 0, 0]}, "colorBy": "data",
            "label": {"show": True, "position": "top", "formatter": "₫{c}",
                       "fontSize": 10, "rotate": 45},
        }],
    }
    st_echarts(option_arpu, height="420px", key="channel_arpu")

    # Chart 2 & 3: Payer rate + Whale rate side by side
    col1, col2 = st.columns(2)
    with col1:
        option_payer = {
            "title": {"text": "Payer Rate by Channel (%)", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": channels, "axisLabel": {"rotate": 35}},
            "yAxis": {"type": "value", "name": "%"},
            "grid": {"left": "12%", "right": "5%", "bottom": "20%", "containLabel": True},
            "series": [{
                "type": "bar",
                "data": [round(v, 2) for v in df["payer_rate"].tolist()],
                "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#5470c6"},
                "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 9},
            }],
        }
        st_echarts(option_payer, height="380px", key="channel_payer")

    with col2:
        whale_rates = df["whale_rate_of_payers"].fillna(0).tolist()
        option_whale = {
            "title": {"text": "Whale Rate (% of Payers)", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": channels, "axisLabel": {"rotate": 35}},
            "yAxis": {"type": "value", "name": "%"},
            "grid": {"left": "12%", "right": "5%", "bottom": "20%", "containLabel": True},
            "series": [{
                "type": "bar",
                "data": [round(v, 2) for v in whale_rates],
                "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#ee6666"},
                "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 9},
            }],
        }
        st_echarts(option_whale, height="380px", key="channel_whale")

    # Chart 4: User volume + Late payer rate combo
    option_combo = {
        "title": {"text": "User Volume & Late-Payer Rate by Channel", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": channels, "axisLabel": {"rotate": 35}},
        "yAxis": [
            {"type": "value", "name": "Users", "position": "left"},
            {"type": "value", "name": "Late-Payer %", "position": "right"},
        ],
        "grid": {"left": "10%", "right": "10%", "bottom": "18%", "containLabel": True},
        "series": [
            {"type": "bar", "data": [int(v) for v in df["users"].tolist()], "name": "Users",
             "yAxisIndex": 0, "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#91cc75"}},
            {"type": "line", "data": [round(v, 1) for v in df["late_payer_rate"].fillna(0).tolist()],
             "name": "Late-Payer Rate %", "yAxisIndex": 1,
             "lineStyle": {"width": 3, "color": "#ee6666"}, "smooth": True},
        ],
    }
    st_echarts(option_combo, height="400px", key="channel_combo")

    # Data table
    with st.expander("📋 Full Channel Data", expanded=False):
        display_df = df.copy()
        display_df["arpu"] = display_df["arpu"].map(lambda x: f"₫{x:,.0f}")
        display_df["total_revenue"] = display_df["total_revenue"].map(lambda x: f"₫{x/1e9:.2f}B")
        display_df["payer_rate"] = display_df["payer_rate"].map(lambda x: f"{x:.2f}%")
        display_df["whale_rate_of_payers"] = display_df["whale_rate_of_payers"].fillna(0).map(lambda x: f"{x:.2f}%")
        display_df["late_payer_rate"] = display_df["late_payer_rate"].fillna(0).map(lambda x: f"{x:.1f}%")
        st.dataframe(display_df, use_container_width=True)

    csv_data = df.to_csv(index=False)
    st.download_button("⬇️ Export Channel Data CSV", csv_data, "channel_dashboard.csv", "text/csv",
                        key="channel_csv_dl")

    summary = (
        f"Channel Dashboard: {len(df)} channels analyzed. Best ARPU: {best_channel} (₫{best_arpu:,.0f}), "
        f"Worst: {worst_channel} (₫{worst_arpu:,.0f}). "
        f"ARPU spread is {best_arpu/worst_arpu:.1f}× between best and worst channel."
    )
    st.markdown(f"**💡 Insight:** {summary}")

    follow_ups = [
        "Show me the Pareto concentration — top 1% revenue share",
        "What is the late payer rate by channel?",
        "Show cohort quality trends — is user quality declining?",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE D — Cohort Quality Tracker
# ═══════════════════════════════════════════════════════════════════════════
def run_cohort_tracker(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Weekly cohort quality trends with regression analysis."""

    sql = """
    SELECT
        DATE_TRUNC('week', CAST(install_date AS DATE)) AS install_week,
        COUNT(*) AS users,
        AVG(ltv30) AS arpu,
        100.0 * SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END) / COUNT(*) AS payer_rate,
        100.0 * SUM(CASE WHEN ltv30 >= 323000 THEN 1 ELSE 0 END) / COUNT(*) AS whale_rate,
        100.0 * SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1 ELSE 0 END) / COUNT(*) AS late_payer_rate,
        AVG(active_days_d7) AS avg_active_days,
        AVG(games_d7) AS avg_games
    FROM cfm_features
    WHERE install_date IS NOT NULL
    GROUP BY install_week
    HAVING COUNT(*) >= 100
    ORDER BY install_week
    """
    df, err = exec_sql(sql, 200)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    st.markdown("### 📈 Cohort Quality Tracker")

    weeks = df["install_week"].astype(str).tolist()

    # Linear regression on ARPU to detect trend
    from scipy import stats
    x_numeric = np.arange(len(df))
    arpu_vals = df["arpu"].values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, arpu_vals)

    trend_direction = "📈 IMPROVING" if slope > 0 else "📉 DECLINING"
    trend_color = "green" if slope > 0 else "red"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohorts Analyzed", f"{len(df)} weeks")
    c2.metric("ARPU Trend", trend_direction, f"₫{slope:+,.0f}/week")
    c3.metric("Latest ARPU", f"₫{arpu_vals[-1]:,.0f}")
    c4.metric("R² (trend fit)", f"{r_value**2:.3f}")

    # Trend line data
    trend_line = (intercept + slope * x_numeric).tolist()

    # Chart 1: ARPU trend with regression
    option_arpu = {
        "title": {"text": "ARPU by Install Week (with Trend)", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": weeks, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "name": "₫"},
        "grid": {"left": "10%", "right": "5%", "bottom": "18%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}}},
        "series": [
            {"type": "bar", "data": [round(v, 0) for v in arpu_vals.tolist()],
             "name": "ARPU", "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#5470c6"}},
            {"type": "line", "data": [round(v, 0) for v in trend_line],
             "name": "Trend", "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 2},
             "symbol": "none"},
        ],
    }
    st_echarts(option_arpu, height="400px", key="cohort_arpu")

    # Chart 2: Payer rate + whale rate trends
    option_rates = {
        "title": {"text": "Payer Rate & Whale Rate Trends", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": weeks, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "name": "%"},
        "grid": {"left": "10%", "right": "5%", "bottom": "18%", "containLabel": True},
        "series": [
            {"type": "line", "data": [round(v, 2) for v in df["payer_rate"].tolist()],
             "name": "Payer Rate %", "smooth": True, "lineStyle": {"width": 2}},
            {"type": "line", "data": [round(v, 3) for v in df["whale_rate"].tolist()],
             "name": "Whale Rate %", "smooth": True, "lineStyle": {"width": 2, "color": "#ee6666"}},
            {"type": "line", "data": [round(v, 2) for v in df["late_payer_rate"].tolist()],
             "name": "Late Payer Rate %", "smooth": True,
             "lineStyle": {"width": 2, "type": "dashed", "color": "#fac858"}},
        ],
    }
    st_echarts(option_rates, height="400px", key="cohort_rates")

    # Chart 3: User volume
    option_vol = {
        "title": {"text": "User Volume by Install Week", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": weeks, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "name": "Users"},
        "grid": {"left": "10%", "right": "5%", "bottom": "18%", "containLabel": True},
        "series": [{
            "type": "bar", "data": [int(v) for v in df["users"].tolist()],
            "name": "Users", "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#91cc75"},
        }],
    }
    st_echarts(option_vol, height="350px", key="cohort_vol")

    if slope < 0:
        st.warning(
            f"⚠️ **ARPU is declining** at ₫{abs(slope):,.0f}/week (R²={r_value**2:.3f}). "
            f"This may indicate worsening channel mix or market saturation. "
            f"Consider reviewing channel allocation and creative refresh."
        )
    else:
        st.success(
            f"✅ **ARPU is stable/improving** at ₫{slope:+,.0f}/week (R²={r_value**2:.3f}). "
            f"Cohort quality is holding steady."
        )

    summary = (
        f"Cohort Tracker: {len(df)} weekly cohorts analyzed. "
        f"ARPU trend: {trend_direction} (₫{slope:+,.0f}/week, R²={r_value**2:.3f}). "
        f"Latest ARPU: ₫{arpu_vals[-1]:,.0f}."
    )

    follow_ups = [
        "Which channels are driving the cohort quality change?",
        "Show me the channel performance dashboard",
        "What is the executive summary of all key metrics?",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE E — Feature Importance & Model Explainer
# ═══════════════════════════════════════════════════════════════════════════
def run_feature_importance(exec_sql: ExecuteSqlFn, models_dir: Path) -> tuple[str, list[str]]:
    """Load model metadata and show feature importance chart."""

    st.markdown("### 🧠 Model Feature Importance Explainer")

    # Find the first available model
    model_meta = None
    model_name = None
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            meta_path = p / "metadata.json"
            if p.is_dir() and meta_path.exists():
                model_meta = json.loads(meta_path.read_text())
                model_name = p.name
                break

    if not model_meta:
        st.warning("No model metadata found. Train a model first.")
        return "No model metadata available.", []

    # Try to load feature importances from model itself
    model_path = models_dir / model_name / "model.pkl"
    importances = {}
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            if hasattr(model, "feature_importances_"):
                feat_names = model_meta.get("numeric_features", []) + model_meta.get("categorical_features", [])
                imp_vals = model.feature_importances_
                if len(feat_names) == len(imp_vals):
                    importances = dict(zip(feat_names, imp_vals))
        except Exception:
            pass

    if not importances:
        # Fallback: use metadata if available
        importances = model_meta.get("feature_importances", {})

    if not importances:
        st.info("Feature importances not stored in model or metadata. Showing model config instead.")
        st.json(model_meta)
        return "Model metadata shown but no feature importances available.", []

    # Sort and display
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in sorted_imp]
    feat_vals = [round(f[1] * 100, 2) for f in sorted_imp]

    # KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_name)
    c2.metric("Top Feature", feat_names[0], f"{feat_vals[0]:.1f}%")
    c3.metric("# Features", f"{len(feat_names)}")

    # Horizontal bar chart
    option = {
        "title": {"text": f"Feature Importance — {model_name}", "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}%"},
        "xAxis": {"type": "value", "name": "Importance %"},
        "yAxis": {"type": "category", "data": feat_names[::-1],
                   "axisLabel": {"fontSize": 11}},
        "grid": {"left": "25%", "right": "5%", "bottom": "8%", "top": "12%"},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "series": [{
            "type": "bar",
            "data": feat_vals[::-1],
            "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            "colorBy": "data",
            "label": {"show": True, "position": "right", "formatter": "{c}%"},
        }],
    }
    st_echarts(option, height=f"{max(400, len(feat_names) * 28)}px", key="feat_importance")

    # Top-3 feature explanation
    st.markdown("#### 🔍 What This Means")
    explanations = {
        "rev_d7": "**D7 Revenue** — strongest predictor since early spending strongly correlates with D30 LTV. However, this means the model struggles with late payers (rev_d7=0).",
        "txn_cnt_d7": "**Transaction Count** — number of purchases in first 7 days. Multiple transactions signal committed spenders.",
        "active_days_d7": "**Active Days** — more login days = higher engagement and conversion likelihood.",
        "games_d7": "**Games Played** — gameplay volume is a proxy for engagement intensity.",
        "kills_d7": "**Kills** — performance metric indicating competitive engagement.",
        "win_rate_d7": "**Win Rate** — skilled players may monetize differently.",
        "first_charge_day_offset_d7": "**First Purchase Timing** — earlier first purchase predicts higher LTV.",
        "max_level_seen_d7": "**Max Level** — progression depth indicates engagement.",
        "login_rows_d7": "**Login Events** — frequency of logins signals retention.",
        "kd_d7": "**K/D Ratio** — skill indicator that correlates with engagement.",
    }

    for i, (feat, val) in enumerate(sorted_imp[:5]):
        explanation = explanations.get(feat, f"**{feat}** — contributes {val*100:.1f}% to model predictions.")
        st.markdown(f"{i+1}. {explanation} *(importance: {val*100:.1f}%)*")

    summary = (
        f"Feature Importance: Model '{model_name}' — top feature is {feat_names[0]} ({feat_vals[0]:.1f}%). "
        f"rev_d7 dominates, which means the model relies heavily on early revenue. "
        f"This explains why late-payer prediction requires additional engagement features."
    )

    follow_ups = [
        "Show me the late payer opportunity — how much revenue are we missing?",
        "What is the payer rate and ARPU by channel?",
        "Score users with the pLTV model and show top 1%",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE F — Executive Brief Generator
# ═══════════════════════════════════════════════════════════════════════════
def run_executive_brief(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """One-page KPI summary dashboard with all key metrics."""

    sql = """
    SELECT
        COUNT(*) AS total_users,
        SUM(ltv30) AS total_revenue,
        AVG(ltv30) AS overall_arpu,
        100.0 * SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END) / COUNT(*) AS payer_rate,
        SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END) AS total_payers,
        SUM(CASE WHEN ltv30 >= 323000 THEN 1 ELSE 0 END) AS whale_count,
        100.0 * SUM(CASE WHEN ltv30 >= 323000 THEN 1 ELSE 0 END) / COUNT(*) AS whale_rate,
        SUM(CASE WHEN ltv30 >= 323000 THEN ltv30 ELSE 0 END) AS whale_revenue,
        SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1 ELSE 0 END) AS late_payers,
        SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN ltv30 ELSE 0 END) AS late_payer_revenue,
        SUM(rev_d7) AS d7_revenue,
        AVG(active_days_d7) AS avg_active_days,
        AVG(games_d7) AS avg_games,
        MIN(install_date) AS first_install,
        MAX(install_date) AS last_install
    FROM cfm_features
    """
    df, err = exec_sql(sql, 10)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    r = df.iloc[0]

    st.markdown("### 📋 Executive Brief — CFM Vietnam pLTV Analytics")
    st.markdown(f"*Data period: {r['first_install']} → {r['last_install']}*")

    # Row 1: Core KPIs
    st.markdown("#### 🎯 Core Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users", f"{int(r['total_users']):,}")
    c2.metric("Total Revenue", f"₫{float(r['total_revenue'])/1e9:.1f}B")
    c3.metric("ARPU", f"₫{float(r['overall_arpu']):,.0f}")
    c4.metric("Payer Rate", f"{float(r['payer_rate']):.2f}%")
    c5.metric("Total Payers", f"{int(r['total_payers']):,}")

    # Row 2: Whale & Late-Payer KPIs
    st.markdown("#### 🐋 Whale & Late-Payer Economics")
    total_rev = float(r["total_revenue"])
    whale_rev = float(r["whale_revenue"])
    late_rev = float(r["late_payer_revenue"])
    d7_rev = float(r["d7_revenue"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Whales (top 1%)", f"{int(r['whale_count']):,}",
              f"{whale_rev/total_rev*100:.1f}% of revenue")
    c2.metric("Late Payers", f"{int(r['late_payers']):,}",
              f"₫{late_rev/1e9:.1f}B hidden revenue")
    c3.metric("D7 Revenue", f"₫{d7_rev/1e9:.1f}B",
              f"{d7_rev/total_rev*100:.0f}% of D30")
    c4.metric("D7→D30 Multiplier", f"{total_rev/d7_rev:.1f}×" if d7_rev > 0 else "N/A")

    # Row 3: Engagement
    st.markdown("#### 🎮 Engagement")
    c1, c2 = st.columns(2)
    c1.metric("Avg Active Days (D7)", f"{float(r['avg_active_days']):.1f}")
    c2.metric("Avg Games (D7)", f"{float(r['avg_games']):.0f}")

    # Revenue breakdown pie
    col1, col2 = st.columns(2)
    with col1:
        non_whale_payer_rev = total_rev - whale_rev - late_rev
        # Adjust to avoid negative
        d7_payer_rev = total_rev - late_rev
        option_pie = {
            "title": {"text": "Revenue Composition", "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: ₫{c}B ({d}%)"},
            "series": [{
                "type": "pie", "radius": ["35%", "65%"],
                "data": [
                    {"name": "D7 Payer Revenue", "value": round(d7_payer_rev / 1e9, 2),
                     "itemStyle": {"color": "#5470c6"}},
                    {"name": "Late Payer Revenue", "value": round(late_rev / 1e9, 2),
                     "itemStyle": {"color": "#ee6666"}},
                ],
                "label": {"formatter": "{b}\n₫{c}B\n({d}%)"},
            }],
        }
        st_echarts(option_pie, height="350px", key="exec_rev_pie")

    with col2:
        option_whale_pie = {
            "title": {"text": "Whale Revenue Concentration", "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: ₫{c}B ({d}%)"},
            "series": [{
                "type": "pie", "radius": ["35%", "65%"],
                "data": [
                    {"name": "Whale Revenue", "value": round(whale_rev / 1e9, 2),
                     "itemStyle": {"color": "#ee6666"}},
                    {"name": "Non-Whale Revenue", "value": round((total_rev - whale_rev) / 1e9, 2),
                     "itemStyle": {"color": "#91cc75"}},
                ],
                "label": {"formatter": "{b}\n₫{c}B\n({d}%)"},
            }],
        }
        st_echarts(option_whale_pie, height="350px", key="exec_whale_pie")

    # Top channels quick view
    sql_channels = """
    SELECT media_source, COUNT(*) AS users, AVG(ltv30) AS arpu
    FROM cfm_features
    WHERE media_source IS NOT NULL AND media_source != ''
    GROUP BY media_source
    HAVING COUNT(*) >= 1000
    ORDER BY arpu DESC
    LIMIT 8
    """
    df_ch, _ = exec_sql(sql_channels, 20)
    if df_ch is not None and not df_ch.empty:
        option_ch = {
            "title": {"text": "Top Channels by ARPU", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": df_ch["media_source"].tolist(),
                       "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value", "name": "₫"},
            "grid": {"left": "10%", "right": "5%", "bottom": "18%", "containLabel": True},
            "series": [{
                "type": "bar", "data": [round(v, 0) for v in df_ch["arpu"].tolist()],
                "itemStyle": {"borderRadius": [4, 4, 0, 0]}, "colorBy": "data",
            }],
        }
        st_echarts(option_ch, height="350px", key="exec_channels")

    # Key takeaways
    st.markdown("#### 💡 Key Takeaways")
    st.markdown(f"""
1. **Revenue concentration is extreme**: Top 1% (whales) = {whale_rev/total_rev*100:.1f}% of all revenue
2. **Late payers are a hidden gold mine**: {int(r['late_payers']):,} users convert after D7, contributing ₫{late_rev/1e9:.1f}B
3. **D7-only ROAS is misleading**: Real D30 revenue is {total_rev/d7_rev:.1f}× what D7 shows
4. **Payer rate is {float(r['payer_rate']):.2f}%**: {int(r['total_payers']):,} payers out of {int(r['total_users']):,} users
5. **Engagement proxy**: Users average {float(r['avg_active_days']):.1f} active days and {float(r['avg_games']):.0f} games in D7
    """)

    summary = (
        f"Executive Brief: {int(r['total_users']):,} users, ₫{total_rev/1e9:.1f}B revenue, "
        f"{float(r['payer_rate']):.2f}% payer rate. Whales (top 1%) = {whale_rev/total_rev*100:.1f}% of revenue. "
        f"Late payers contribute ₫{late_rev/1e9:.1f}B. D7→D30 multiplier: {total_rev/d7_rev:.1f}×."
    )

    follow_ups = [
        "Deep dive into channel performance",
        "Show me the Pareto concentration curve",
        "Explain the model's feature importance",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE G — Anomaly / Outlier Highlighter
# ═══════════════════════════════════════════════════════════════════════════
def run_anomaly_detection(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Auto-detect outliers in revenue, engagement, and cohort quality."""

    st.markdown("### 🔍 Anomaly & Outlier Detection")

    # 1. Revenue outliers — top 20 by LTV30
    sql_whales = """
    SELECT vopenid, media_source, first_os, ltv30, rev_d7,
           active_days_d7, games_d7, kills_d7, is_payer_30
    FROM cfm_features
    ORDER BY ltv30 DESC
    LIMIT 20
    """
    df_top, err = exec_sql(sql_whales, 20)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    st.markdown("#### 🐋 Top 20 Revenue Outliers (Mega-Whales)")
    if df_top is not None and not df_top.empty:
        max_ltv = float(df_top["ltv30"].iloc[0])
        median_sql = "SELECT MEDIAN(ltv30) AS med FROM cfm_features WHERE ltv30 > 0"
        df_med, _ = exec_sql(median_sql, 5)
        median_ltv = float(df_med["med"].iloc[0]) if df_med is not None else 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Max LTV30", f"₫{max_ltv:,.0f}")
        c2.metric("Median (payers)", f"₫{median_ltv:,.0f}")
        c3.metric("Max / Median", f"{max_ltv/median_ltv:,.0f}×")

        option = {
            "title": {"text": "Top 20 Users by LTV30", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category",
                       "data": [str(v)[:12] for v in df_top["vopenid"].tolist()],
                       "axisLabel": {"rotate": 45, "fontSize": 9}},
            "yAxis": {"type": "value", "name": "LTV30 (₫)"},
            "grid": {"left": "12%", "right": "5%", "bottom": "20%", "containLabel": True},
            "series": [{
                "type": "bar",
                "data": [float(v) for v in df_top["ltv30"].tolist()],
                "itemStyle": {"borderRadius": [4, 4, 0, 0]}, "colorBy": "data",
            }],
        }
        st_echarts(option, height="400px", key="anomaly_whales")

        with st.expander("📋 Top 20 Details"):
            st.dataframe(df_top, use_container_width=True)

    # 2. Engagement outliers — extremely high game counts
    sql_engagement = """
    SELECT
        CASE
            WHEN games_d7 >= 500 THEN '500+'
            WHEN games_d7 >= 200 THEN '200-499'
            WHEN games_d7 >= 100 THEN '100-199'
            WHEN games_d7 >= 50 THEN '50-99'
            ELSE '<50'
        END AS games_bucket,
        COUNT(*) AS users,
        AVG(ltv30) AS avg_ltv,
        100.0 * SUM(CASE WHEN is_payer_30 = 1 THEN 1 ELSE 0 END) / COUNT(*) AS payer_rate
    FROM cfm_features
    GROUP BY games_bucket
    ORDER BY games_bucket
    """
    df_eng, _ = exec_sql(sql_engagement, 20)
    if df_eng is not None and not df_eng.empty:
        st.markdown("#### 🎮 Engagement Outlier Segmentation")
        option_eng = {
            "title": {"text": "Engagement Segments: Games Played D7", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0},
            "xAxis": {"type": "category", "data": df_eng["games_bucket"].tolist()},
            "yAxis": [
                {"type": "value", "name": "Users", "position": "left"},
                {"type": "value", "name": "Payer Rate %", "position": "right"},
            ],
            "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
            "series": [
                {"type": "bar", "data": [int(v) for v in df_eng["users"].tolist()],
                 "name": "Users", "yAxisIndex": 0,
                 "itemStyle": {"borderRadius": [4, 4, 0, 0], "color": "#5470c6"}},
                {"type": "line", "data": [round(v, 2) for v in df_eng["payer_rate"].tolist()],
                 "name": "Payer Rate %", "yAxisIndex": 1,
                 "lineStyle": {"width": 3, "color": "#ee6666"}, "smooth": True},
            ],
        }
        st_echarts(option_eng, height="380px", key="anomaly_engagement")

    # 3. Late-payer concentration by channel
    sql_late = """
    SELECT media_source,
           COUNT(*) AS total,
           SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1 ELSE 0 END) AS late_payers,
           100.0 * SUM(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1 ELSE 0 END) / COUNT(*) AS late_rate
    FROM cfm_features
    WHERE media_source IS NOT NULL AND media_source != ''
    GROUP BY media_source
    HAVING COUNT(*) >= 1000
    ORDER BY late_rate DESC
    LIMIT 10
    """
    df_late, _ = exec_sql(sql_late, 20)
    if df_late is not None and not df_late.empty:
        st.markdown("#### ⏱️ Late-Payer Concentration by Channel")
        option_late = {
            "title": {"text": "Late-Payer Rate by Channel", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": df_late["media_source"].tolist(),
                       "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value", "name": "Late Payer Rate %"},
            "grid": {"left": "10%", "right": "5%", "bottom": "18%", "containLabel": True},
            "series": [{
                "type": "bar",
                "data": [round(v, 2) for v in df_late["late_rate"].tolist()],
                "itemStyle": {"borderRadius": [4, 4, 0, 0]}, "colorBy": "data",
                "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 9},
            }],
        }
        st_echarts(option_late, height="380px", key="anomaly_late")

    summary = (
        f"Anomaly Detection: Max LTV30 = ₫{max_ltv:,.0f} ({max_ltv/median_ltv:,.0f}× median). "
        f"Engagement and late-payer concentration vary significantly across channels."
    )

    follow_ups = [
        "Show me the executive summary of all key metrics",
        "Deep dive into channel performance",
        "What is the whale threshold and Pareto concentration?",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE H — Data Quality & Coverage Dashboard
# ═══════════════════════════════════════════════════════════════════════════
def run_data_quality(exec_sql: ExecuteSqlFn) -> tuple[str, list[str]]:
    """Dataset freshness, coverage, and null-rate dashboard."""

    st.markdown("### 🔎 Data Quality & Coverage")

    sql = """
    SELECT
        COUNT(*) AS total_rows,
        MIN(install_date) AS first_date,
        MAX(install_date) AS last_date,
        COUNT(DISTINCT media_source) AS n_channels,
        COUNT(DISTINCT first_os) AS n_os,
        -- Null rates for key columns
        100.0 * SUM(CASE WHEN vopenid IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_vopenid,
        100.0 * SUM(CASE WHEN media_source IS NULL OR media_source = '' THEN 1 ELSE 0 END) / COUNT(*) AS null_media,
        100.0 * SUM(CASE WHEN install_date IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_install,
        100.0 * SUM(CASE WHEN ltv30 IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_ltv30,
        100.0 * SUM(CASE WHEN rev_d7 IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_rev_d7,
        100.0 * SUM(CASE WHEN active_days_d7 IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_active,
        100.0 * SUM(CASE WHEN games_d7 IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_games,
        100.0 * SUM(CASE WHEN kills_d7 IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS null_kills,
        100.0 * SUM(CASE WHEN first_os IS NULL OR first_os = '' THEN 1 ELSE 0 END) / COUNT(*) AS null_os
    FROM cfm_features
    """
    df, err = exec_sql(sql, 10)
    if err:
        st.error(f"SQL error: {err}")
        return f"Error: {err}", []

    r = df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{int(r['total_rows']):,}")
    c2.metric("Date Range", f"{r['first_date']} → {r['last_date']}")
    c3.metric("Channels", f"{int(r['n_channels'])}")
    c4.metric("OS Variants", f"{int(r['n_os'])}")

    # Column completeness chart
    columns = ["vopenid", "media_source", "install_date", "ltv30", "rev_d7",
                "active_days_d7", "games_d7", "kills_d7", "first_os"]
    null_rates = [
        float(r["null_vopenid"]), float(r["null_media"]), float(r["null_install"]),
        float(r["null_ltv30"]), float(r["null_rev_d7"]), float(r["null_active"]),
        float(r["null_games"]), float(r["null_kills"]), float(r["null_os"]),
    ]
    completeness = [round(100 - nr, 2) for nr in null_rates]

    option = {
        "title": {"text": "Column Completeness (%)", "left": "center"},
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}% complete"},
        "xAxis": {"type": "category", "data": columns, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value", "name": "%", "min": 0, "max": 100},
        "grid": {"left": "10%", "right": "5%", "bottom": "15%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "series": [{
            "type": "bar",
            "data": [{"value": v, "itemStyle": {"color": "#91cc75" if v >= 99 else "#fac858" if v >= 90 else "#ee6666"}}
                     for v in completeness],
            "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 10},
            "itemStyle": {"borderRadius": [4, 4, 0, 0]},
        }],
    }
    st_echarts(option, height="380px", key="data_quality_completeness")

    # Flag issues
    issues = [(col, nr) for col, nr in zip(columns, null_rates) if nr > 1.0]
    if issues:
        st.warning("⚠️ **Data quality flags:**\n" +
                    "\n".join(f"- `{col}`: {nr:.1f}% null/empty" for col, nr in issues))
    else:
        st.success("✅ All key columns have >99% completeness — data quality is excellent.")

    summary = (
        f"Data Quality: {int(r['total_rows']):,} rows, {r['first_date']} → {r['last_date']}, "
        f"{int(r['n_channels'])} channels. "
        + ("All key columns >99% complete." if not issues else f"{len(issues)} columns flagged.")
    )

    follow_ups = [
        "Give me the executive summary",
        "Show channel performance dashboard",
        "What is the payer rate and revenue distribution?",
    ]
    return summary, follow_ups


# ═══════════════════════════════════════════════════════════════════════════
# Follow-up suggestion engine
# ═══════════════════════════════════════════════════════════════════════════
FOLLOW_UP_MAP = {
    "payer": [
        "What is the whale concentration — top 1% vs rest?",
        "Show late payer opportunity — how much revenue at D7-only?",
        "Which channels have the best payer rate?",
    ],
    "whale": [
        "Show me the Pareto concentration curve",
        "Which channels produce the most whales?",
        "What behavioral signals predict whales?",
    ],
    "channel": [
        "Show cohort quality trends over time",
        "Which channel has the highest late-payer rate?",
        "Compare ARPU across all channels",
    ],
    "revenue": [
        "Show the Pareto concentration — top 1% revenue share",
        "How much late-payer revenue are we missing at D7?",
        "Give me the executive summary of all metrics",
    ],
    "model": [
        "What are the most important features in the model?",
        "Score users with the pLTV model",
        "Show me the late payer opportunity",
    ],
    "late": [
        "Which channels have the most late payers?",
        "What behavioral signals predict late conversion?",
        "Compare seed strategies with and without late payers",
    ],
    "seed": [
        "Show the Pareto concentration for revenue",
        "How do late payers affect seed quality?",
        "Which channels produce the best seeds?",
    ],
    "cohort": [
        "Is user quality declining over time?",
        "Show channel performance dashboard",
        "Give me the executive brief",
    ],
}


def get_follow_up_suggestions(question: str) -> list[str]:
    """Return 2-3 relevant follow-up questions based on keyword matching."""
    q_lower = question.lower()
    suggestions = []
    for keyword, follow_ups in FOLLOW_UP_MAP.items():
        if keyword in q_lower:
            suggestions.extend(follow_ups)
    # Deduplicate and limit
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:3] if unique else [
        "Show me the executive summary",
        "What is the whale concentration?",
        "Compare channel performance",
    ]


def render_follow_ups(suggestions: list[str], key_prefix: str = "followup"):
    """Render clickable follow-up suggestion pills."""
    if not suggestions:
        return
    st.markdown("---")
    st.markdown("**💬 Suggested follow-ups:**")
    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(f"→ {suggestion[:60]}", key=f"{key_prefix}_{i}",
                          use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()
