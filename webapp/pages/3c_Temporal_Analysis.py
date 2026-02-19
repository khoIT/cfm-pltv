"""
Page 3c — Temporal Analysis
Interactive cohort-level analysis of user quality over time.
Allows dataset selection and reruns analysis dynamically.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, render_top_menu, render_report_md, get_data, convert_vnd, get_currency_info,
    format_currency, DATA_DIR, REPORTS_DIR,
)

render_top_menu()
render_sidebar()


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing temporal metrics…")
def compute_temporal_metrics(csv_path: str, file_mtime: float):
    """Load a CSV and compute daily cohort metrics."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "install_date" not in df.columns or "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'install_date' and 'ltv30' columns."

    df["install_date"] = pd.to_datetime(df["install_date"])
    df["is_payer_d7"] = (df["rev_d7"] > 0).astype(int) if "rev_d7" in df.columns else 0
    df["is_payer_30"] = (df.get("is_payer_30", df["ltv30"] > 0)).astype(int)
    df["is_late_payer"] = ((df.get("rev_d7", pd.Series(0, index=df.index)) == 0) & (df["ltv30"] > 0)).astype(int)
    df["rev_d7"] = df.get("rev_d7", pd.Series(0.0, index=df.index)).astype(float)

    agg_dict = {
        "users": ("ltv30", "count"),
        "payer_rate_d30": ("is_payer_30", "mean"),
        "payer_rate_d7": ("is_payer_d7", "mean"),
        "late_payer_rate": ("is_late_payer", "mean"),
        "mean_ltv30": ("ltv30", "mean"),
        "median_ltv30": ("ltv30", "median"),
        "total_ltv30": ("ltv30", "sum"),
        "total_rev_d7": ("rev_d7", "sum"),
    }
    # Add engagement columns if present
    if "games_d7" in df.columns:
        agg_dict["mean_games"] = ("games_d7", "mean")
    if "active_days_d7" in df.columns:
        agg_dict["mean_active_days"] = ("active_days_d7", "mean")
    if "login_rows_d7" in df.columns:
        agg_dict["mean_logins"] = ("login_rows_d7", "mean")
    if "kd_d7" in df.columns:
        agg_dict["mean_kd"] = ("kd_d7", "mean")

    daily = df.groupby("install_date").agg(**agg_dict).reset_index()
    daily["arpu_d30"] = daily["total_ltv30"] / daily["users"]
    daily["arpu_d7"] = daily["total_rev_d7"] / daily["users"]
    daily["d7_d30_ratio"] = np.where(daily["total_ltv30"] > 0, daily["total_rev_d7"] / daily["total_ltv30"], 0)
    daily["install_dow"] = daily["install_date"].dt.day_name()

    # Flag cohorts younger than 15 days — their D30 metrics are incomplete
    today = pd.Timestamp.now().normalize()
    daily["days_since_install"] = (today - daily["install_date"]).dt.days
    daily["d30_mature"] = daily["days_since_install"] >= 15

    # Null out D30 metrics for immature cohorts so they don't distort charts/aggregates
    immature_mask = ~daily["d30_mature"]
    for col in ["arpu_d30", "mean_ltv30", "median_ltv30", "total_ltv30",
                "payer_rate_d30", "late_payer_rate", "d7_d30_ratio"]:
        if col in daily.columns:
            daily.loc[immature_mask, col] = np.nan

    return df, daily, None


def list_available_datasets():
    """List CSV files in the data directory."""
    datasets = {}
    for f in DATA_DIR.glob("cfm_pltv*.csv"):
        size_mb = f.stat().st_size / 1e6
        datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    return datasets


# ── page ───────────────────────────────────────────────────────────
st.title("📈 Temporal Analysis")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page.")
    st.stop()

cur = get_currency_info()

st.markdown(
    "Analyze how **user quality evolves** over install cohorts. "
    "Track payer rates, ARPU, engagement, and the D7→D30 revenue gap across time."
)

# =====================================================================
# DATASET SELECTOR
# =====================================================================
st.header("📂 Select Dataset")
datasets = list_available_datasets()

if not datasets:
    st.error("No datasets found in data/ directory.")
    st.stop()

ds_names = list(datasets.keys())
default_idx = ds_names.index("cfm_pltv") if "cfm_pltv" in ds_names else 0

col_ds1, col_ds2 = st.columns([2, 3])
with col_ds1:
    chosen_ds = st.selectbox(
        "Dataset", ds_names, index=default_idx, key="temporal_dataset",
        help="Choose which dataset to analyze"
    )
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.1f} MB")

# Load & compute
ds_path = ds_info["path"]
ds_mtime = ds_info["mtime"]
df_raw, daily, error = compute_temporal_metrics(ds_path, ds_mtime)

if error:
    st.error(f"❌ {error}")
    st.stop()

n_users = len(df_raw)
n_dates = daily["install_date"].nunique()
date_min = daily["install_date"].min().date()
date_max = daily["install_date"].max().date()

st.success(f"✅ Loaded **{n_users:,}** users across **{n_dates}** install dates ({date_min} → {date_max})")

n_immature = (~daily["d30_mature"]).sum()
if n_immature > 0:
    immature_dates = daily.loc[~daily["d30_mature"], "install_date"].dt.strftime("%Y-%m-%d").tolist()
    st.warning(
        f"⚠️ **{n_immature} cohort(s) are less than 15 days old** and have incomplete D30 data. "
        f"Their D30 metrics (ARPU, payer rate, LTV30) are hidden to avoid misleading results.  \n"
        f"Affected dates: `{', '.join(immature_dates)}`"
    )

# ── Report reference ──────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Temporal_Analysis.md", "📄 Full Temporal Analysis Report")

# =====================================================================
# KPI SUMMARY
# =====================================================================
st.markdown("---")
st.header("📊 Cohort KPIs")

# Only use mature cohorts (>=15 days) for D30 aggregate KPIs
mature_dates = daily.loc[daily["d30_mature"], "install_date"]
df_mature = df_raw[df_raw["install_date"].isin(mature_dates)]
n_mature = len(df_mature)

total_revenue = daily["total_ltv30"].sum()  # NaN cohorts excluded by nansum default
avg_payer_rate = df_mature["is_payer_30"].mean() if n_mature > 0 else np.nan
avg_late_rate = df_mature["is_late_payer"].mean() if n_mature > 0 else np.nan
overall_arpu = total_revenue / n_mature if n_mature > 0 else 0
d7_ratio = daily["total_rev_d7"].sum() / total_revenue if total_revenue > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Users", f"{n_users:,}")
with kpi2:
    st.metric("D30 Payer Rate", f"{avg_payer_rate:.2%}")
with kpi3:
    st.metric("Late Payer Rate", f"{avg_late_rate:.2%}")
with kpi4:
    st.metric("Overall ARPU", format_currency(overall_arpu, cur["code"]))
with kpi5:
    st.metric("D7/D30 Rev Ratio", f"{d7_ratio:.1%}",
              help="How much of D30 revenue is captured by D7")

# =====================================================================
# CHART 1: Volume + Payer Rates
# =====================================================================
st.markdown("---")
st.header("📉 Daily Cohort Trends")

col_vol, col_arpu = st.columns(2)

with col_vol:
    st.subheader("Install Volume & Payer Rates")
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Bar(x=daily["install_date"], y=daily["users"], name="Users",
               marker_color="#FF6600", opacity=0.6),
        secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=daily["install_date"], y=daily["payer_rate_d30"]*100,
                   name="Payer Rate D30 (%)", line=dict(color="royalblue", width=3)),
        secondary_y=True
    )
    fig1.add_trace(
        go.Scatter(x=daily["install_date"], y=daily["payer_rate_d7"]*100,
                   name="Payer Rate D7 (%)", line=dict(color="#e74c3c", width=2, dash="dash")),
        secondary_y=True
    )
    fig1.add_trace(
        go.Scatter(x=daily["install_date"], y=daily["late_payer_rate"]*100,
                   name="Late Payer Rate (%)", line=dict(color="#2ecc71", width=2, dash="dot")),
        secondary_y=True
    )
    fig1.update_layout(height=420, legend=dict(orientation="h", y=-0.25), margin=dict(b=80))
    fig1.update_yaxes(title_text="Users", secondary_y=False)
    fig1.update_yaxes(title_text="Rate (%)", secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

# =====================================================================
# CHART 2: ARPU Trends
# =====================================================================
with col_arpu:
    st.subheader("ARPU Trends (D7 vs D30)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily["install_date"], y=convert_vnd(daily["arpu_d30"], cur["code"]),
        name="ARPU D30", line=dict(color="royalblue", width=3),
        hovertemplate="Date: %{x}<br>ARPU D30: %{y:,.0f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=daily["install_date"], y=convert_vnd(daily["arpu_d7"], cur["code"]),
        name="ARPU D7", line=dict(color="#e74c3c", width=2, dash="dash"),
        hovertemplate="Date: %{x}<br>ARPU D7: %{y:,.0f}<extra></extra>",
    ))
    fig2.update_layout(
        yaxis_title=f"ARPU ({cur['symbol']})", height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# CHART 3: D7/D30 Revenue Ratio + Engagement
# =====================================================================
st.markdown("---")
col_ratio, col_eng = st.columns(2)

with col_ratio:
    st.subheader("D7 Revenue as % of D30")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=daily["install_date"], y=daily["d7_d30_ratio"]*100,
        marker_color="#FF6600",
        hovertemplate="Date: %{x}<br>D7/D30: %{y:.1f}%<extra></extra>",
    ))
    fig3.add_hline(y=daily["d7_d30_ratio"].mean()*100, line_dash="dash", line_color="gray",
                   annotation_text=f"Avg: {daily['d7_d30_ratio'].mean()*100:.1f}%")
    fig3.update_layout(yaxis_title="D7 / D30 Revenue (%)", height=400)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"On average, D7 captures **{daily['d7_d30_ratio'].mean()*100:.1f}%** of D30 revenue. "
               f"The remaining **{(1-daily['d7_d30_ratio'].mean())*100:.1f}%** accrues D8–D30.")

with col_eng:
    st.subheader("Engagement Trends")
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    if "mean_games" in daily.columns:
        fig4.add_trace(
            go.Scatter(x=daily["install_date"], y=daily["mean_games"],
                       name="Avg Games D7", line=dict(color="#9b59b6", width=2)),
            secondary_y=False
        )
    if "mean_active_days" in daily.columns:
        fig4.add_trace(
            go.Scatter(x=daily["install_date"], y=daily["mean_active_days"],
                       name="Avg Active Days D7", line=dict(color="#1abc9c", width=2)),
            secondary_y=True
        )
    fig4.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    fig4.update_yaxes(title_text="Games", secondary_y=False)
    fig4.update_yaxes(title_text="Active Days", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
# CHART 5: Cumulative Revenue Build-Up
# =====================================================================
st.markdown("---")
st.header("💰 Revenue Build-Up Over Time")

daily_sorted = daily.sort_values("install_date")
daily_sorted["cum_users"] = daily_sorted["users"].cumsum()
daily_sorted["cum_ltv30"] = daily_sorted["total_ltv30"].cumsum()
daily_sorted["cum_rev_d7"] = daily_sorted["total_rev_d7"].cumsum()

col_cum, col_table = st.columns([1.3, 1])

with col_cum:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=daily_sorted["install_date"],
        y=convert_vnd(daily_sorted["cum_ltv30"], cur["code"]),
        name="Cumulative LTV30", fill="tozeroy",
        line=dict(color="royalblue", width=2),
    ))
    fig5.add_trace(go.Scatter(
        x=daily_sorted["install_date"],
        y=convert_vnd(daily_sorted["cum_rev_d7"], cur["code"]),
        name="Cumulative Rev D7", fill="tozeroy",
        line=dict(color="#e74c3c", width=2),
    ))
    fig5.update_layout(
        title="Cumulative Revenue by Install Date",
        yaxis_title=f"Revenue ({cur['symbol']})", height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_table:
    st.subheader("Daily Cohort Summary")
    table_df = daily_sorted[["install_date", "users", "payer_rate_d30", "late_payer_rate", "arpu_d30", "d7_d30_ratio"]].copy()
    table_df.columns = ["Date", "Users", "D30 Payer %", "Late Payer %", f"ARPU ({cur['symbol']})", "D7/D30 %"]
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    table_df["D30 Payer %"] = (table_df["D30 Payer %"] * 100).round(2)
    table_df["Late Payer %"] = (table_df["Late Payer %"] * 100).round(2)
    table_df[f"ARPU ({cur['symbol']})"] = table_df[f"ARPU ({cur['symbol']})"].apply(
        lambda v: format_currency(v, cur["code"])
    )
    table_df["D7/D30 %"] = (table_df["D7/D30 %"] * 100).round(1)
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=420)

# =====================================================================
# CHART 6: Day-of-Week Analysis (if enough data)
# =====================================================================
if n_dates >= 7:
    st.markdown("---")
    st.header("📅 Day-of-Week Patterns")

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_sorted["dow"] = pd.Categorical(daily_sorted["install_dow"], categories=dow_order, ordered=True)
    dow_agg = daily_sorted.groupby("dow", observed=True).agg(
        avg_users=("users", "mean"),
        avg_payer_rate=("payer_rate_d30", "mean"),
        avg_arpu=("arpu_d30", "mean"),
    ).reset_index()

    col_dow1, col_dow2 = st.columns(2)
    with col_dow1:
        fig_dow1 = px.bar(dow_agg, x="dow", y="avg_users", title="Avg Users by Day of Week",
                          color_discrete_sequence=["#FF6600"])
        fig_dow1.update_layout(height=350, xaxis_title="", yaxis_title="Avg Users")
        st.plotly_chart(fig_dow1, use_container_width=True)
    with col_dow2:
        fig_dow2 = px.bar(dow_agg, x="dow", y="avg_payer_rate",
                          title="Avg Payer Rate by Day of Week",
                          color_discrete_sequence=["royalblue"])
        fig_dow2.update_layout(height=350, xaxis_title="", yaxis_title="Payer Rate",
                               yaxis_tickformat=".2%")
        st.plotly_chart(fig_dow2, use_container_width=True)

# =====================================================================
# INSIGHTS
# =====================================================================
st.markdown("---")
st.header("💡 Insights")

# Compute insights dynamically
first_day = daily_sorted.iloc[0]
last_day = daily_sorted.iloc[-1]
peak_users_day = daily_sorted.loc[daily_sorted["users"].idxmax()]
peak_payer_day = daily_sorted.loc[daily_sorted["payer_rate_d30"].idxmax()]

insights = []

# Volume trend
vol_change = (last_day["users"] - first_day["users"]) / first_day["users"]
if abs(vol_change) > 0.3:
    direction = "📉 declining" if vol_change < 0 else "📈 growing"
    insights.append(f"Install volume is **{direction}** ({vol_change:+.0%} from first to last day)")

# Payer rate stability
payer_std = daily_sorted["payer_rate_d30"].std()
payer_mean = daily_sorted["payer_rate_d30"].mean()
cv = payer_std / payer_mean if payer_mean > 0 else 0
if cv < 0.1:
    insights.append(f"D30 payer rate is **stable** across cohorts (CV={cv:.2f})")
elif cv < 0.3:
    insights.append(f"D30 payer rate shows **moderate variation** across cohorts (CV={cv:.2f})")
else:
    insights.append(f"⚠️ D30 payer rate is **highly variable** across cohorts (CV={cv:.2f}) — investigate root cause")

# D7/D30 gap
avg_ratio = daily_sorted["d7_d30_ratio"].mean()
insights.append(f"D7 captures only **{avg_ratio:.1%}** of D30 revenue — "
                f"**{1-avg_ratio:.1%}** accrues after D7, validating late payer detection")

# Late payer consistency
late_std = daily_sorted["late_payer_rate"].std()
late_mean = daily_sorted["late_payer_rate"].mean()
insights.append(f"Late payer rate averages **{late_mean:.2%}** (σ={late_std:.4f}) — "
                f"{'consistent' if late_std/late_mean < 0.15 else 'variable'} across cohorts")

# Peak day
insights.append(f"Peak install day: **{peak_users_day['install_date'].strftime('%Y-%m-%d')}** "
                f"with {int(peak_users_day['users']):,} users")
insights.append(f"Best payer rate day: **{peak_payer_day['install_date'].strftime('%Y-%m-%d')}** "
                f"at {peak_payer_day['payer_rate_d30']:.2%}")

for insight in insights:
    st.markdown(f"- {insight}")

# Business recommendations
st.markdown("### 🎯 Recommended Actions")
if avg_ratio < 0.7:
    st.success("✅ **Late payer revenue is significant.** ML-based late payer detection can capture "
               f"the {1-avg_ratio:.0%} of revenue that D7 heuristics miss.")
if cv > 0.2:
    st.warning("⚠️ **Cohort quality varies significantly.** Implement weekly quality monitoring "
               "and adjust UA budgets based on recent cohort performance.")
if vol_change < -0.5:
    st.info("📉 **Install volume declining rapidly.** This is expected post-launch. "
            "Monitor whether paid UA can sustain quality as organic fades.")
