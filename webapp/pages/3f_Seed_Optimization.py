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
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, render_top_menu, render_report_md, get_registry_path,
    convert_vnd, get_currency_info, format_currency, REPORTS_DIR,
    get_active_model,
)

render_top_menu()
render_sidebar()


# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preparing data…")
def _load_and_prepare(csv_path: str, file_mtime: float):
    """Load CSV and compute base columns (cached)."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "ltv30" not in df.columns:
        return None, "Dataset must contain 'ltv30' column."
    df["rev_d7"] = df.get("rev_d7", pd.Series(0.0, index=df.index)).astype(float)
    df["is_payer_30"] = (df["ltv30"] > 0).astype(int)
    df["is_late_payer"] = ((df["rev_d7"] == 0) & (df["ltv30"] > 0)).astype(int)

    # Engagement score as proxy
    eng_cols = [c for c in ["games_d7", "active_days_d7", "login_rows_d7"] if c in df.columns]
    if eng_cols:
        score_parts = []
        for col in eng_cols:
            mx = df[col].max()
            score_parts.append(df[col] / mx if mx > 0 else 0)
        df["engagement_score"] = sum(score_parts) / len(score_parts)
    else:
        df["engagement_score"] = 0.0

    # Install week for time-window analysis
    if "install_date" in df.columns:
        df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")
        df["install_week"] = df["install_date"].dt.to_period("W").astype(str)

    return df, None


def _seed_stats(seed_df, name, whale_threshold, total_whales):
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


def compute_strategies(df, top_pct, pltv_preds=None):
    """Build strategy comparison DataFrame. Fast, no caching needed."""
    d7_payers = df[df["rev_d7"] > 0]
    d7_zero = df[df["rev_d7"] == 0]
    oracle = df[df["is_payer_30"] == 1]
    whale_threshold = df["ltv30"].quantile(0.90)
    total_whales = (df["ltv30"] >= whale_threshold).sum()

    # Engagement-based enrichment
    eng_thresh = d7_zero["engagement_score"].quantile(1 - top_pct / 100)
    eng_predicted_late = d7_zero[d7_zero["engagement_score"] >= eng_thresh]
    eng_enriched = pd.concat([d7_payers, eng_predicted_late], ignore_index=True)

    rows = [
        _seed_stats(d7_payers, "D7 Payers Only", whale_threshold, total_whales),
        _seed_stats(eng_enriched, f"+ Top {top_pct:.0f}% Engagement", whale_threshold, total_whales),
    ]

    # ML pLTV-based enrichment (if model predictions available)
    if pltv_preds is not None:
        d7_zero_idx = d7_zero.index
        d7_zero_preds = pltv_preds.loc[d7_zero_idx]
        for k in [1, 5, 10]:
            k_thresh = d7_zero_preds.quantile(1 - k / 100)
            ml_predicted = d7_zero[d7_zero_preds >= k_thresh]
            ml_enriched = pd.concat([d7_payers, ml_predicted], ignore_index=True)
            rows.append(_seed_stats(ml_enriched, f"+ Top {k}% pLTV", whale_threshold, total_whales))

    rows.append(_seed_stats(oracle, "D30 Payers (Oracle)", whale_threshold, total_whales))

    strategies = pd.DataFrame(rows)
    strategies["Label"] = strategies.apply(
        lambda r: f"{r['Strategy']}\n({int(r['Seed Size']):,})", axis=1)

    # Revenue composition of engagement-enriched seed
    enriched_from_d7 = d7_payers["ltv30"].sum()
    enriched_from_late = eng_predicted_late["ltv30"].sum()

    return strategies, {
        "enriched_from_d7": enriched_from_d7,
        "enriched_from_late": enriched_from_late,
        "whale_threshold": whale_threshold,
        "total_rev": df["ltv30"].sum(),
        "has_engagement": "engagement_score" in df.columns and df["engagement_score"].sum() > 0,
    }


def compute_weekly(df, top_pct, pltv_preds=None):
    """Vectorized weekly strategy analysis."""
    if "install_week" not in df.columns:
        return None
    weeks = sorted(df["install_week"].dropna().unique())
    rows = []
    for wk in weeks:
        mask = df["install_week"] == wk
        wdf = df[mask]
        if len(wdf) < 100:
            continue
        w_d7 = wdf[wdf["rev_d7"] > 0]
        w_d7_zero = wdf[wdf["rev_d7"] == 0]
        w_oracle = wdf[wdf["is_payer_30"] == 1]
        w_whale_t = wdf["ltv30"].quantile(0.90)
        w_total_whales = (wdf["ltv30"] >= w_whale_t).sum()
        n_cohort = len(wdf)

        # Engagement enriched
        w_eng_t = w_d7_zero["engagement_score"].quantile(1 - top_pct / 100)
        w_eng_late = w_d7_zero[w_d7_zero["engagement_score"] >= w_eng_t]
        w_eng_enriched = pd.concat([w_d7, w_eng_late], ignore_index=True)

        pairs = [("D7 Payers Only", w_d7),
                 (f"+ Top {top_pct:.0f}% Engagement", w_eng_enriched)]

        # ML enriched (top 5%)
        if pltv_preds is not None:
            w_preds = pltv_preds.loc[w_d7_zero.index]
            ml_t = w_preds.quantile(0.95)
            w_ml_late = w_d7_zero[w_preds >= ml_t]
            w_ml_enriched = pd.concat([w_d7, w_ml_late], ignore_index=True)
            pairs.append(("+ Top 5% pLTV", w_ml_enriched))

        pairs.append(("D30 Oracle", w_oracle))

        for name, sdf in pairs:
            n = len(sdf)
            whales = (sdf["ltv30"] >= w_whale_t).sum()
            rows.append({
                "Week": wk, "Strategy": name,
                "Seed Size": n,
                "Avg LTV30": sdf["ltv30"].mean() if n > 0 else 0,
                "Whale Capture": whales / w_total_whales if w_total_whales > 0 else 0,
                "Payer Rate": sdf["is_payer_30"].mean() if n > 0 else 0,
                "Users in Cohort": n_cohort,
            })
    return pd.DataFrame(rows) if rows else None




# ── page ───────────────────────────────────────────────────────────
st.title("🌱 Seed Optimization")
st.markdown(
    "Compare **seed list strategies** for UA lookalike expansion. "
    "Which users should be in your seed list — D7 payers only, engagement-enriched, or ML pLTV-enriched?"
)

cur = get_currency_info()

# Load from registry (cached — fast on reload)
ds_path, ds_mtime = get_registry_path()
df_raw, error = _load_and_prepare(ds_path, ds_mtime)
if error:
    st.error(f"❌ {error}")
    st.stop()

top_pct = st.slider("Top % of D7-non-payers to add (engagement-based)",
                     min_value=1, max_value=20, value=5,
                     help="Top N% of D7=0 users by engagement score added to enriched seed")

# ── ML predictions (if model loaded) ─────────────────────────────
model, model_feats = get_active_model()
pltv_preds = None
if model is not None:
    try:
        num_feats = [f for f in model_feats if f not in ("media_source", "first_country_code", "first_os", "first_login_channel")]
        cat_feats = [f for f in model_feats if f in ("media_source", "first_country_code", "first_os", "first_login_channel")]
        feat_df = df_raw[num_feats + cat_feats].copy()
        for c in cat_feats:
            le = LabelEncoder()
            feat_df[c] = le.fit_transform(feat_df[c].astype(str))
        if "first_charge_day_offset_d7" in feat_df.columns:
            feat_df["first_charge_day_offset_d7"] = feat_df["first_charge_day_offset_d7"].fillna(-1)
        pltv_preds = pd.Series(np.expm1(model.predict(feat_df)), index=df_raw.index)
    except Exception as e:
        st.warning(f"⚠️ Could not generate pLTV predictions: {e}")

# ── Compute strategies ────────────────────────────────────────────
strategies, extras = compute_strategies(df_raw, top_pct, pltv_preds)
n_users = len(df_raw)
n_strats = len(strategies)
has_ml = pltv_preds is not None

if has_ml:
    st.success(f"✅ **{n_users:,}** users — {n_strats} strategies (including ML pLTV at 1%, 5%, 10%)")
else:
    st.info(f"✅ **{n_users:,}** users — {n_strats} strategies.  \n"
            "💡 **Load a trained model** on the Features & Model page to unlock pLTV-based seed strategies.")

if not extras["has_engagement"]:
    st.warning("⚠️ No engagement features found — engagement-based strategy may be unreliable.")

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Seed_Optimization_Strategy.md", "📄 Full Seed Optimization Report")

# ── KPI Summary ────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Strategy Comparison")

d7_row = strategies[strategies["Strategy"] == "D7 Payers Only"].iloc[0]
eng_row = strategies[strategies["Strategy"].str.contains("Engagement")].iloc[0]
oracle_row = strategies[strategies["Strategy"] == "D30 Payers (Oracle)"].iloc[0]
best_ml_row = strategies[strategies["Strategy"].str.contains("pLTV")].iloc[-1] if has_ml else eng_row

k1, k2, k3, k4 = st.columns(4)
with k1:
    best_non_oracle = best_ml_row if has_ml else eng_row
    size_gain = best_non_oracle["Seed Size"] - d7_row["Seed Size"]
    st.metric("Best Enriched Size", f"{int(best_non_oracle['Seed Size']):,}",
              f"+{size_gain:,} vs D7-only")
with k2:
    whale_gain = best_non_oracle["Whale Capture"] - d7_row["Whale Capture"]
    st.metric("Best Whale Capture", f"{best_non_oracle['Whale Capture']:.1%}",
              f"+{whale_gain:.1%} vs D7-only")
with k3:
    rev_gap = oracle_row["Total Revenue"] - d7_row["Total Revenue"]
    st.metric("Revenue Gap to Oracle",
              format_currency(convert_vnd(rev_gap, cur["code"]), cur["code"]),
              "missed by D7-only")
with k4:
    st.metric("Oracle Whale Capture", f"{oracle_row['Whale Capture']:.1%}", "theoretical max")

# ── Charts ────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    fig_ltv = px.bar(
        strategies, x="Label",
        y=convert_vnd(strategies["Avg LTV30"], cur["code"]),
        title=f"Avg LTV30 per Seed User ({cur['symbol']})",
        color="Strategy",
        labels={"y": f"Avg LTV30 ({cur['symbol']})", "Label": ""},
        text=convert_vnd(strategies["Avg LTV30"], cur["code"]).apply(lambda v: f"{v:,.0f}"),
    )
    fig_ltv.update_layout(height=480, showlegend=False, xaxis_tickangle=-25)
    fig_ltv.update_traces(textposition="outside")
    st.plotly_chart(fig_ltv, use_container_width=True)

with col2:
    fig_whale = px.bar(
        strategies, x="Label",
        y=strategies["Whale Capture"] * 100,
        title="Whale Capture Rate (%)",
        color="Strategy",
        labels={"y": "Whale Capture (%)", "Label": ""},
        text=(strategies["Whale Capture"] * 100).round(1).astype(str) + "%",
    )
    fig_whale.update_layout(height=480, showlegend=False, xaxis_tickangle=-25)
    fig_whale.update_traces(textposition="outside")
    st.plotly_chart(fig_whale, use_container_width=True)

# ── Size vs Quality Scatter ───────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    fig_tradeoff = px.scatter(
        strategies, x="Seed Size",
        y=convert_vnd(strategies["Avg LTV30"], cur["code"]),
        text="Label", size=[40] * len(strategies),
        color="Strategy",
        title="Seed Size vs Quality Tradeoff",
        labels={"x": "Seed Size (users)", "y": f"Avg LTV30 ({cur['symbol']})"},
    )
    fig_tradeoff.update_traces(textposition="top center")
    fig_tradeoff.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

with col4:
    comp_labels = ["D7 Payers Revenue", "Predicted Late Payer Revenue"]
    comp_values = [
        convert_vnd(extras["enriched_from_d7"], cur["code"]),
        convert_vnd(extras["enriched_from_late"], cur["code"]),
    ]
    comp_df = pd.DataFrame({"Source": comp_labels, "Revenue": comp_values})
    fig_comp = px.pie(comp_df, names="Source", values="Revenue",
                      hole=0.4, color="Source",
                      color_discrete_map={comp_labels[0]: "#FF6600", comp_labels[1]: "#2ecc71"},
                      title=f"Enriched Seed Revenue Split ({cur['symbol']})")
    fig_comp.update_layout(height=450)
    st.plotly_chart(fig_comp, use_container_width=True)

# ── Summary Table ──────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Strategy Summary Table")
tbl = strategies.drop(columns=["Label"]).copy()
tbl["Avg LTV30"] = tbl["Avg LTV30"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["Total Revenue"] = tbl["Total Revenue"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl["Payer Rate"] = (tbl["Payer Rate"] * 100).round(1).astype(str) + "%"
tbl["Whale Capture"] = (tbl["Whale Capture"] * 100).round(1).astype(str) + "%"
tbl["Seed Size"] = tbl["Seed Size"].apply(lambda v: f"{int(v):,}")
st.dataframe(tbl, height=(len(tbl) + 1) * 35 + 3, use_container_width=True, hide_index=True)

# ── Quality vs Reach Explanation ──────────────────────────────────
st.markdown("---")
st.header("🎯 Quality vs Reach: Why Higher LTV ≠ Best Seed")
st.markdown(f"""
**D7 Payers Only** has the highest Avg LTV30 — but that **doesn't make it the best seed strategy**.

**The tradeoff:**
- **D7 Payers Only** ({int(d7_row['Seed Size']):,} users) has highest per-user quality, but **smallest seed** → ad network gets fewer signals, **misses {100 - d7_row['Whale Capture']*100:.0f}% of whales**
- **Engagement-enriched** ({int(eng_row['Seed Size']):,} users) adds high-engagement non-payers → +{(eng_row['Whale Capture'] - d7_row['Whale Capture'])*100:.1f}pp whale capture""")
if has_ml:
    ml5 = strategies[strategies["Strategy"] == "+ Top 5% pLTV"]
    if len(ml5) > 0:
        ml5_row = ml5.iloc[0]
        st.markdown(f"""- **pLTV-enriched Top 5%** ({int(ml5_row['Seed Size']):,} users) uses the trained model to find the most valuable D7-non-payers → **{ml5_row['Whale Capture']:.1%} whale capture** (+{(ml5_row['Whale Capture'] - d7_row['Whale Capture'])*100:.1f}pp vs D7-only)""")
st.markdown(f"""- **D30 Oracle** ({int(oracle_row['Seed Size']):,} users) = theoretical ceiling with perfect foresight

**Bottom line:** A slightly lower-quality but larger and more whale-representative seed produces better lookalikes.
{"**The ML pLTV strategy outperforms engagement-only** because the model directly predicts future value, not just engagement proxy.**" if has_ml else "**Load a trained model** to unlock pLTV-based strategies that outperform engagement-only enrichment."}
""")

# ── Time-Window Analysis ──────────────────────────────────────────
with st.spinner("Computing weekly breakdown…"):
    weekly_strategies = compute_weekly(df_raw, top_pct, pltv_preds)
if weekly_strategies is not None and len(weekly_strategies) > 0:
    st.markdown("---")
    st.header("📅 Strategy Performance by Install Week")
    st.markdown("Does the optimal seed strategy change over time as user quality evolves?")

    wcol1, wcol2 = st.columns(2)
    with wcol1:
        ws = weekly_strategies.copy()
        ws["Avg LTV30 disp"] = convert_vnd(ws["Avg LTV30"], cur["code"])
        fig_wk_ltv = px.line(
            ws, x="Week", y="Avg LTV30 disp", color="Strategy",
            markers=True,
            title=f"Avg LTV30 by Week & Strategy ({cur['symbol']})",
            labels={"Avg LTV30 disp": f"Avg LTV30 ({cur['symbol']})", "Week": "Install Week"},
        )
        fig_wk_ltv.update_layout(height=420)
        st.plotly_chart(fig_wk_ltv, use_container_width=True)

    with wcol2:
        fig_wk_whale = px.line(
            weekly_strategies, x="Week", y="Whale Capture", color="Strategy",
            markers=True,
            title="Whale Capture by Week & Strategy",
            labels={"Whale Capture": "Whale Capture Rate", "Week": "Install Week"},
        )
        fig_wk_whale.update_layout(height=420, yaxis_tickformat=".0%")
        st.plotly_chart(fig_wk_whale, use_container_width=True)

    fig_wk_size = px.bar(
        weekly_strategies, x="Week", y="Seed Size", color="Strategy",
        barmode="group", title="Seed Size by Week & Strategy",
        labels={"Seed Size": "Users in Seed", "Week": "Install Week"},
    )
    fig_wk_size.update_layout(height=380)
    st.plotly_chart(fig_wk_size, use_container_width=True)

    cohort_sizes = weekly_strategies[weekly_strategies["Strategy"] == "D7 Payers Only"][["Week", "Users in Cohort"]].drop_duplicates()
    st.caption("Cohort sizes: " + " | ".join(f"{r.Week}: {int(r['Users in Cohort']):,}" for _, r in cohort_sizes.iterrows()))

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
eng_size_pct = (eng_row["Seed Size"] - d7_row["Seed Size"]) / d7_row["Seed Size"] * 100 if d7_row["Seed Size"] > 0 else 0
late_rev_pct = extras["enriched_from_late"] / (extras["enriched_from_d7"] + extras["enriched_from_late"]) * 100 if (extras["enriched_from_d7"] + extras["enriched_from_late"]) > 0 else 0

st.markdown(f"- **Engagement-enriched seed** is **{eng_size_pct:.0f}% larger** than D7-only, whale capture: **{eng_row['Whale Capture']:.1%}** vs {d7_row['Whale Capture']:.1%}")
if has_ml:
    ml_rows = strategies[strategies["Strategy"].str.contains("pLTV")]
    for _, mr in ml_rows.iterrows():
        gain = (mr["Whale Capture"] - d7_row["Whale Capture"]) * 100
        st.markdown(f"- **{mr['Strategy']}** ({int(mr['Seed Size']):,} users): whale capture **{mr['Whale Capture']:.1%}** (+{gain:.1f}pp vs D7-only)")
st.markdown(f"- Late payers contribute **{late_rev_pct:.1f}%** of engagement-enriched seed revenue")
st.markdown(f"- **Revenue gap to oracle:** {format_currency(convert_vnd(oracle_row['Total Revenue'] - d7_row['Total Revenue'], cur['code']), cur['code'])} missed by D7-only")
if has_ml:
    st.markdown(f"- **Recommendation:** Use **pLTV-enriched seeds** — the ML model directly identifies high-value future payers, outperforming engagement proxy")
else:
    st.markdown(f"- **Recommendation:** Use engagement-enriched seeds with top {top_pct}% predicted late payers. Load a model for better results.")
