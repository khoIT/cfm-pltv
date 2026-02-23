"""
Page 3i — Time-to-First-Purchase Analysis
Survival curve, conversion timing segments, and LTV30 by first-charge day.
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


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Analysing first-purchase timing…")
def compute_ttp_metrics(csv_path: str, file_mtime: float):
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, None, "Dataset must contain 'ltv30' column."
    if "first_charge_day_offset_d7" not in df.columns:
        return None, None, "Dataset must contain 'first_charge_day_offset_d7' column."

    df = df.copy()
    df["ltv30"] = pd.to_numeric(df["ltv30"], errors="coerce").fillna(0)
    df["first_charge_day_offset_d7"] = pd.to_numeric(
        df["first_charge_day_offset_d7"], errors="coerce")

    payers = df[df["ltv30"] > 0].copy()
    if len(payers) == 0:
        return None, None, "No payers found in dataset."

    # Survival curve: % of payers who have charged by day D
    max_day = 7
    survival = []
    for d in range(0, max_day + 1):
        converted = payers[payers["first_charge_day_offset_d7"] <= d]
        pct = len(converted) / len(payers) * 100
        survival.append({"Day": d, "% Payers Converted": round(pct, 1),
                         "Cumulative Payers": len(converted)})
    survival_df = pd.DataFrame(survival)

    # Timing segments
    def timing_seg(v):
        if pd.isna(v):    return "D8+ / Unknown"
        if v == 0:        return "D0 (same-day)"
        if v <= 3:        return "D1–D3"
        return "D4–D7"

    seg_order = ["D0 (same-day)", "D1–D3", "D4–D7", "D8+ / Unknown"]
    payers["timing_seg"] = payers["first_charge_day_offset_d7"].apply(timing_seg)
    payers["timing_seg"] = pd.Categorical(payers["timing_seg"], categories=seg_order, ordered=True)

    seg_stats = payers.groupby("timing_seg", observed=True).agg(
        users=("ltv30", "count"),
        avg_ltv30=("ltv30", "mean"),
        median_ltv30=("ltv30", "median"),
        total_ltv30=("ltv30", "sum"),
    ).reset_index()
    seg_stats["pct_payers"] = (seg_stats["users"] / len(payers) * 100).round(1)
    seg_stats["rev_share"] = (seg_stats["total_ltv30"] / payers["ltv30"].sum() * 100).round(1)

    # LTV30 by first-charge day (D0–D7)
    day_ltv = payers[payers["first_charge_day_offset_d7"].between(0, 7)].groupby(
        "first_charge_day_offset_d7").agg(
        users=("ltv30", "count"),
        avg_ltv30=("ltv30", "mean"),
        median_ltv30=("ltv30", "median"),
    ).reset_index()
    day_ltv.columns = ["Day", "Users", "Avg LTV30", "Median LTV30"]

    # Engagement profile by timing segment
    eng_cols = [c for c in ["games_d7", "active_days_d7", "win_rate_d7", "kd_d7",
                             "max_level_seen_d7"] if c in payers.columns]
    eng_profile = None
    if eng_cols:
        eng_profile = payers.groupby("timing_seg", observed=True)[eng_cols].mean().reset_index()

    # Whale rate by timing segment
    p95 = df["ltv30"].quantile(0.95)
    payers["is_whale"] = (payers["ltv30"] >= p95).astype(int)
    whale_by_seg = payers.groupby("timing_seg", observed=True)["is_whale"].mean().reset_index()
    whale_by_seg.columns = ["timing_seg", "whale_rate"]
    whale_by_seg["whale_rate_pct"] = (whale_by_seg["whale_rate"] * 100).round(2)

    return df, {
        "payers": payers,
        "survival_df": survival_df,
        "seg_stats": seg_stats,
        "day_ltv": day_ltv,
        "eng_profile": eng_profile,
        "whale_by_seg": whale_by_seg,
        "n_payers": len(payers),
        "n_total": len(df),
        "p95": p95,
    }, None


# ── page ───────────────────────────────────────────────────────────
st.title("⏱️ Time-to-First-Purchase")
st.markdown(
    "When do users make their **first purchase**? Does converting earlier predict higher LTV30? "
    "Identify the optimal intervention window for monetization nudges."
)

cur = get_currency_info()

# Load from registry
ds_path, ds_mtime = get_registry_path()
with st.spinner("Analysing first-purchase timing…"):
    df_raw, metrics, error = compute_ttp_metrics(ds_path, ds_mtime)

if error:
    st.error(f"❌ {error}")
    st.stop()

survival_df = metrics["survival_df"]
seg_stats = metrics["seg_stats"]
day_ltv = metrics["day_ltv"]
eng_profile = metrics["eng_profile"]
whale_by_seg = metrics["whale_by_seg"]
n_payers = metrics["n_payers"]
n_total = metrics["n_total"]
p95 = metrics["p95"]

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Time_to_First_Purchase.md", "📄 Full Time-to-First-Purchase Report")

# ── KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Key Metrics")
st.info(
    """💡 **Why does first-purchase timing matter?**

In F2P games, **when** a user makes their first purchase is one of the strongest predictors of their lifetime value.
Users who pay on Day 0 (same session as install) typically have **2–5× higher LTV** than those who wait until D4–D7.

This page helps you answer three critical questions:
1. **When is the "golden window"?** — The optimal time to show a first-purchase offer.
2. **Are late converters worth less?** — If D4–D7 payers have much lower LTV, your offers are working but attracting low-intent buyers.
3. **Where are the whales?** — Which conversion window produces the most top spenders?""",
    icon="⏱️"
)

d0_pct = seg_stats[seg_stats["timing_seg"] == "D0 (same-day)"]["pct_payers"].values
d0_pct = d0_pct[0] if len(d0_pct) > 0 else 0
d3_cum = survival_df[survival_df["Day"] == 3]["% Payers Converted"].values
d3_cum = d3_cum[0] if len(d3_cum) > 0 else 0
d0_avg = seg_stats[seg_stats["timing_seg"] == "D0 (same-day)"]["avg_ltv30"].values
d0_avg = d0_avg[0] if len(d0_avg) > 0 else 0
d47_avg = seg_stats[seg_stats["timing_seg"] == "D4–D7"]["avg_ltv30"].values
d47_avg = d47_avg[0] if len(d47_avg) > 0 else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Payer Rate (D30)", f"{n_payers/n_total*100:.1f}%", f"{n_payers:,} payers")
with k2:
    st.metric("D0 Converters", f"{d0_pct:.1f}%", "of all payers")
with k3:
    st.metric("Converted by D3", f"{d3_cum:.1f}%", "cumulative")
with k4:
    ratio = d0_avg / d47_avg if d47_avg > 0 else 0
    st.metric("D0 vs D4–D7 Avg LTV30", f"{ratio:.1f}×", "D0 payers are higher value")

# ── Survival Curve ─────────────────────────────────────────────────
st.markdown("---")
st.header("📈 Payer Conversion Survival Curve")
st.markdown(
    "**Left chart:** Shows cumulative % of payers who have made their first purchase by each day. "
    "The steeper the curve rises early, the more your game drives **impulse monetization**. "
    "The 80% line marks when you've captured most conversions. "
    "**Right chart:** Shows what % of all payers fall into each conversion window."
)
col1, col2 = st.columns(2)

with col1:
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=survival_df["Day"], y=survival_df["% Payers Converted"],
        mode="lines+markers+text",
        text=survival_df["% Payers Converted"].astype(str) + "%",
        textposition="top center",
        line=dict(color="#3498db", width=3),
        marker=dict(size=10),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
        name="Cumulative % Converted",
    ))
    fig_surv.add_hline(y=80, line_dash="dash", line_color="orange",
                       annotation_text="80% threshold")
    fig_surv.update_layout(
        title="Cumulative % of Payers Converted by Day",
        xaxis_title="Days Since Install",
        yaxis_title="% of Payers Converted",
        xaxis=dict(tickvals=list(range(8))),
        height=400,
    )
    st.plotly_chart(fig_surv, use_container_width=True)

with col2:
    fig_seg = go.Figure()
    colors = ["#e74c3c", "#e67e22", "#3498db", "#95a5a6"]
    for i, row in seg_stats.iterrows():
        fig_seg.add_trace(go.Bar(
            x=[str(row["timing_seg"])],
            y=[row["pct_payers"]],
            name=str(row["timing_seg"]),
            marker_color=colors[i % len(colors)],
            text=[f"{row['pct_payers']:.1f}%"],
            textposition="outside",
        ))
    fig_seg.update_layout(
        title="% of Payers by Conversion Window",
        yaxis_title="% of Payers", height=400,
        showlegend=False, yaxis=dict(range=[0, 60]),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

# ── LTV30 by Timing ──────────────────────────────────────
st.markdown("---")
st.header("💰 LTV30 by First-Purchase Timing")
st.markdown(
    "**Left chart:** Average LTV30 by conversion window. If D0 is dramatically higher, it confirms that "
    "same-day converters are inherently more motivated spenders — not just responding to offers. "
    "**Right chart:** Day-by-day LTV decline shows exactly when value drops off, helping you set "
    "the optimal deadline for first-purchase promotions."
)
col3, col4 = st.columns(2)

with col3:
    fig_ltv_seg = go.Figure()
    for i, row in seg_stats.iterrows():
        avg_disp = convert_vnd(row["avg_ltv30"], cur["code"])
        fig_ltv_seg.add_trace(go.Bar(
            x=[str(row["timing_seg"])],
            y=[avg_disp],
            name=str(row["timing_seg"]),
            marker_color=colors[i % len(colors)],
            text=[format_currency(avg_disp, cur["code"])],
            textposition="outside",
        ))
    fig_ltv_seg.update_layout(
        title=f"Avg LTV30 by Conversion Window ({cur['symbol']})",
        yaxis_title=f"Avg LTV30 ({cur['symbol']})", height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_ltv_seg, use_container_width=True)

with col4:
    if len(day_ltv) > 0:
        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(
            x=day_ltv["Day"], y=day_ltv["Avg LTV30"].apply(
                lambda v: convert_vnd(v, cur["code"])),
            mode="lines+markers",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=10),
            name="Avg LTV30",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_ltv["Day"], y=day_ltv["Median LTV30"].apply(
                lambda v: convert_vnd(v, cur["code"])),
            mode="lines+markers",
            line=dict(color="#3498db", width=2, dash="dash"),
            marker=dict(size=8),
            name="Median LTV30",
        ))
        fig_day.update_layout(
            title=f"Avg & Median LTV30 by First-Charge Day ({cur['symbol']})",
            xaxis_title="First Charge Day",
            yaxis_title=f"LTV30 ({cur['symbol']})",
            xaxis=dict(tickvals=list(range(8))),
            height=400,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_day, use_container_width=True)

# ── Whale Rate by Timing ──────────────────────────────────
st.markdown("---")
st.header("🐋 Whale Rate by Conversion Window")
st.info(
    """💡 **Why whale rate by timing matters:**

Not all payers are equal. This chart shows what % of payers in each conversion window become **whales** (top 5% LTV30).

- **D0 whale rate >> D4–D7 whale rate:** Confirms that early converters are your whale pipeline. Protect this segment.
- **If D4–D7 has similar whale rate:** Your later-stage offers are successfully converting high-value users — keep them.
- **If D4–D7 whale rate ≈ 0:** Late converters are almost never whales. Offers past D3 mainly capture low-value impulse buys.

**Engagement chart (right):** Shows whether late converters play differently. If D4–D7 has low games/active_days, 
they are likely responding to a push notification — not genuinely engaged.""",
    icon="🐋"
)
col5, col6 = st.columns(2)

with col5:
    fig_whale = go.Figure()
    for i, row in whale_by_seg.iterrows():
        fig_whale.add_trace(go.Bar(
            x=[str(row["timing_seg"])],
            y=[row["whale_rate_pct"]],
            marker_color=colors[i % len(colors)],
            text=[f"{row['whale_rate_pct']:.2f}%"],
            textposition="outside",
        ))
    fig_whale.update_layout(
        title="Whale Rate (top 5% LTV30) by Conversion Window",
        yaxis_title="Whale Rate (%)", height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_whale, use_container_width=True)

with col6:
    if eng_profile is not None:
        eng_cols_disp = [c for c in eng_profile.columns if c != "timing_seg"]
        fig_eng = go.Figure()
        eng_colors = ["#e74c3c", "#e67e22", "#3498db", "#95a5a6"]
        for i, row in eng_profile.iterrows():
            fig_eng.add_trace(go.Bar(
                name=str(row["timing_seg"]),
                x=[c.replace("_d7", "") for c in eng_cols_disp],
                y=[row[c] for c in eng_cols_disp],
                marker_color=eng_colors[i % len(eng_colors)],
            ))
        fig_eng.update_layout(
            title="Avg Engagement by Conversion Window",
            barmode="group", height=380,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_eng, use_container_width=True)

# ── Summary Table ───────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Segment Summary")
tbl = seg_stats.copy()
tbl["avg_ltv30"] = tbl["avg_ltv30"].apply(
    lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["median_ltv30"] = tbl["median_ltv30"].apply(
    lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl.columns = ["Segment", "Users", f"Avg LTV30 ({cur['symbol']})",
               f"Median LTV30 ({cur['symbol']})", "Total LTV30",
               "% of Payers", "Rev Share %"]
st.dataframe(tbl[["Segment", "Users", "% of Payers", f"Avg LTV30 ({cur['symbol']})",
                   f"Median LTV30 ({cur['symbol']})", "Rev Share %"]],
             use_container_width=True, hide_index=True)

# ── Monetization Opportunity ─────────────────────────────
st.markdown("---")
st.header("💵 Monetization Opportunity Sizing")

# Calculate opportunity: what if we could shift 10% of D4-D7 payers to D0-D3 behavior?
d03_avg = seg_stats[seg_stats["timing_seg"].isin(["D0 (same-day)", "D1–D3"])]["avg_ltv30"].mean()
d47_users = seg_stats[seg_stats["timing_seg"] == "D4–D7"]["users"].values
d47_users = d47_users[0] if len(d47_users) > 0 else 0
shift_users = int(d47_users * 0.1)
incremental_rev = shift_users * (d03_avg - d47_avg) if d47_avg > 0 else 0

opp_col1, opp_col2, opp_col3 = st.columns(3)
with opp_col1:
    st.metric("🎯 Golden Window", "D0–D3",
              f"{d3_cum:.0f}% of conversions happen here")
with opp_col2:
    non_payer_count = n_total - n_payers
    st.metric("Non-Payers (D30)", f"{non_payer_count:,}",
              f"{non_payer_count/n_total*100:.1f}% of all users")
with opp_col3:
    st.metric("If 10% of D4–D7 shifted to D0–D3",
              format_currency(convert_vnd(incremental_rev, cur["code"]), cur["code"]),
              f"{shift_users:,} users × higher LTV")

st.markdown(
    f"> **Opportunity:** There are **{non_payer_count:,} non-payers** in the dataset. "
    f"Even converting **1% more** at D0–D3 LTV would add "
    f"**{format_currency(convert_vnd(int(non_payer_count * 0.01) * d03_avg, cur['code']), cur['code'])}** "
    f"in incremental revenue. The first-purchase offer timing is your biggest monetization lever."
)

# ── Insights & Playbook ───────────────────────────────────
st.markdown("---")
st.header("💡 Key Findings")

find_col1, find_col2 = st.columns(2)
with find_col1:
    st.markdown(f"""
**Conversion Timing:**
- **{d0_pct:.1f}%** of payers convert on D0 — arrive with purchase intent
- **{d3_cum:.1f}%** have converted by D3 — validates D3 as primary scoring window
- D0 payers have **{ratio:.1f}×** higher avg LTV30 than D4–D7 payers
""")
with find_col2:
    st.markdown("""
**Behavioral Patterns:**
- D4–D7 converters are likely nudged by offers — lower intrinsic motivation = lower LTV
- Late converters have lower engagement (fewer games, fewer active days)
- D0 conversion rate is the best single proxy for UA campaign quality
""")

st.markdown("### 🎯 First-Purchase Optimization Playbook")
play_col1, play_col2, play_col3 = st.columns(3)
with play_col1:
    st.markdown("""
**📣 D0: Impulse Window**
- Show a **starter pack** after the 3rd game in session 1
- Use price anchoring (show a premium pack first, then a "value" pack)
- D0 converters are your future whales — don't discount too aggressively
""")
with play_col2:
    st.markdown("""
**⏰ D1–D3: Engagement Window**
- Send push notification to high-engagement non-payers on D2
- Offer a **limited-time** first-purchase bonus (expires D3)
- Personalize: skilled players get competitive items, casual players get cosmetics
""")
with play_col3:
    st.markdown("""
**🚨 D4–D7: Last Chance Window**
- These users are unlikely to become whales — optimize for **any conversion**
- Deep discount (50%+ off) is acceptable here since LTV is already low
- Use D0 conversion rate as your primary **UA campaign quality signal**
""")
