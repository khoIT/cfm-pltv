"""
Page 3g — Real-Time Scoring (Early Prediction)
Uses actual D1/D3/D5/D7 window data from cfm_pltv_D135 dataset.
Each window_days value in the dataset contains real feature aggregations
for that number of days post-install.
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

# Feature columns present in the D135 multi-window dataset
WINDOW_FEATURE_COLS = [
    "login_rows", "active_days", "loginchannel_variety", "network_variety",
    "clientversion_variety", "max_level_seen", "max_viplevel_seen", "max_ladderscore",
    "games", "win_rate", "avg_game_duration", "avg_score",
    "kills", "deaths", "assists", "kd", "max_level_game", "max_ladderlevel",
    "rev", "txn_cnt",
]

# ── helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training models on actual D1/D3/D5/D7 windows…")
def compute_realtime_metrics(csv_path: str, file_mtime: float):
    df = pd.read_csv(csv_path, low_memory=False)

    required = {"window_days", "ltv30"}
    missing = required - set(df.columns)
    if missing:
        return None, None, (
            f"Dataset missing columns: {missing}. "
            "Please select a cfm_pltv_D135_part*.csv file which contains actual multi-window data."
        )

    feat_cols = [c for c in WINDOW_FEATURE_COLS if c in df.columns]
    if not feat_cols:
        return None, None, "No feature columns found. Expected login_rows, games, rev, etc."

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr

    windows_present = sorted(df["window_days"].unique())
    window_map = {1: "D1", 3: "D3", 5: "D5", 7: "D7"}

    results = []
    preds_by_window = {}
    y_true_d7 = None

    # Use D7 data as the label source (ltv30 is the same for all windows per user)
    df_d7 = df[df["window_days"] == 7].copy()
    if len(df_d7) == 0:
        return None, None, "No D7 rows found in dataset. Cannot establish baseline."

    # Build a vopenid → ltv30 map from D7 rows
    ltv_map = df_d7.set_index("vopenid")["ltv30"].to_dict()
    y_d7 = df_d7["ltv30"].values
    y_true_d7 = y_d7

    # Train/test split on D7 users (same users across all windows)
    vopenids_d7 = df_d7["vopenid"].values
    X_d7 = df_d7[feat_cols].fillna(0).values
    X_train, X_test, y_train, y_test, vid_train, vid_test = train_test_split(
        X_d7, y_d7, vopenids_d7, test_size=0.2, random_state=42
    )
    vid_test_set = set(vid_test)

    for w_days in windows_present:
        if w_days not in window_map:
            continue
        window_name = window_map[w_days]
        df_w = df[df["window_days"] == w_days].copy()

        # Align test users: use same test vopenids as D7 split
        df_w_test = df_w[df_w["vopenid"].isin(vid_test_set)].copy()
        df_w_train = df_w[~df_w["vopenid"].isin(vid_test_set)].copy()

        if len(df_w_train) < 100 or len(df_w_test) < 100:
            continue

        X_w_train = df_w_train[feat_cols].fillna(0).values
        y_w_train = df_w_train["ltv30"].values
        X_w_test = df_w_test[feat_cols].fillna(0).values
        y_w_test = df_w_test["ltv30"].values

        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=42
        )
        model.fit(X_w_train, np.log1p(y_w_train))
        y_pred = np.expm1(model.predict(X_w_test))
        y_pred = np.clip(y_pred, 0, None)

        n = len(y_w_test)
        spearman_rho, _ = spearmanr(y_w_test, y_pred)
        rmse = np.sqrt(np.mean((y_w_test - y_pred) ** 2))
        top10_idx = np.argsort(y_pred)[::-1][:max(1, int(n * 0.10))]
        lift = y_w_test[top10_idx].sum() / y_w_test.sum() if y_w_test.sum() > 0 else 0

        results.append({
            "Window": window_name,
            "Days": w_days,
            "Spearman ρ": round(spearman_rho, 4),
            "Lift@10%": round(lift * 100, 1),
            "RMSE": round(rmse, 0),
            "Test Users": n,
        })
        preds_by_window[window_name] = (y_w_test, y_pred)

    if not results:
        return None, None, "Could not compute metrics for any window."

    results_df = pd.DataFrame(results).sort_values("Days")

    d7_spearman = results_df.loc[results_df["Window"] == "D7", "Spearman ρ"].values
    d7_lift = results_df.loc[results_df["Window"] == "D7", "Lift@10%"].values
    if len(d7_spearman) > 0 and d7_spearman[0] > 0:
        results_df["Spearman Retention %"] = (results_df["Spearman ρ"] / d7_spearman[0] * 100).round(1)
        results_df["Lift Retention %"] = (results_df["Lift@10%"] / d7_lift[0] * 100).round(1)
    else:
        results_df["Spearman Retention %"] = 100.0
        results_df["Lift Retention %"] = 100.0

    n_unique_users = df["vopenid"].nunique()
    return df, {
        "results": results_df,
        "preds": preds_by_window,
        "y_true_d7": y_true_d7,
        "feat_cols": feat_cols,
        "n_unique_users": n_unique_users,
    }, None


def list_available_datasets():
    """List D135 part files first, then other cfm_pltv datasets."""
    datasets = {}
    # Prioritise D135 parts
    for f in sorted(DATA_DIR.glob("cfm_pltv_D135_part*.csv")):
        size_mb = f.stat().st_size / 1e6
        datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    # Then other datasets
    for f in DATA_DIR.glob("cfm_pltv*.csv"):
        if f.stem not in datasets:
            size_mb = f.stat().st_size / 1e6
            datasets[f.stem] = {"path": str(f), "size_mb": size_mb, "mtime": f.stat().st_mtime}
    return datasets


# ── page ───────────────────────────────────────────────────────────
st.title("⚡ Real-Time Scoring")
st.markdown(
    "Evaluate **D1 / D3 / D5 / D7** prediction windows using **actual behavioural data** "
    "aggregated at each window length. Quantifies how much accuracy is lost by predicting earlier, "
    "enabling faster seed generation, bid adjustments, and campaign kill decisions."
)
st.success(
    "✅ **Using real multi-window data** (Dec 16 2025 – Feb 19 2026). "
    "Each window uses actual feature aggregations — no simulation.",
    icon="✅"
)

cur = get_currency_info()

# ── Dataset selector ───────────────────────────────────────────────
st.header("📂 Select Dataset Part")
datasets = list_available_datasets()
if not datasets:
    st.error("No datasets found in data/ directory.")
    st.stop()

ds_names = list(datasets.keys())
# Default to part01
default_idx = next((i for i, n in enumerate(ds_names) if "D135_part01" in n), 0)
col_ds1, col_ds2 = st.columns([2, 3])
with col_ds1:
    chosen_ds = st.selectbox(
        "Dataset part", ds_names, index=default_idx, key="realtime_dataset",
        help="Each part contains ~1M rows across D1/D3/D5/D7 windows (~250k unique users)"
    )
with col_ds2:
    ds_info = datasets[chosen_ds]
    st.markdown(f"**{chosen_ds}** — {ds_info['size_mb']:.0f} MB")
    st.caption("Contains actual D1/D3/D5/D7 feature windows per user")

with st.spinner("Training GBM models on actual D1/D3/D5/D7 windows… (~20–40s, cached after first run)"):
    df_raw, metrics, error = compute_realtime_metrics(ds_info["path"], ds_info["mtime"])

if error:
    st.error(f"❌ {error}")
    st.stop()

results_df = metrics["results"]
preds = metrics["preds"]
n_unique = metrics["n_unique_users"]
n_feat = len(metrics["feat_cols"])

st.success(f"✅ Evaluated **{n_unique:,}** unique users across {len(results_df)} prediction windows using {n_feat} features")

# ── Report ─────────────────────────────────────────────────────────
render_report_md(REPORTS_DIR / "Real_Time_Scoring.md", "📄 Full Real-Time Scoring Report")

# ── KPI Summary ────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Window Performance Summary")

d3_rows = results_df[results_df["Window"] == "D3"]
d7_rows = results_df[results_df["Window"] == "D7"]
d1_rows = results_df[results_df["Window"] == "D1"]
d3_row = d3_rows.iloc[0] if len(d3_rows) > 0 else results_df.iloc[-1]
d7_row = d7_rows.iloc[0] if len(d7_rows) > 0 else results_df.iloc[-1]
d1_row = d1_rows.iloc[0] if len(d1_rows) > 0 else results_df.iloc[0]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("D3 Spearman ρ", f"{d3_row['Spearman ρ']:.4f}",
              f"{d3_row['Spearman Retention %']:.1f}% of D7")
with k2:
    st.metric("D3 Lift@10%", f"{d3_row['Lift@10%']:.1f}%",
              f"{d3_row['Lift Retention %']:.1f}% of D7")
with k3:
    st.metric("D1 Retention", f"{d1_row['Spearman Retention %']:.1f}%",
              "6 days earlier than D7")
with k4:
    st.metric("D7 Lift@10%", f"{d7_row['Lift@10%']:.1f}%", "baseline")

# ── Chart 1: Metrics by Window ─────────────────────────────────────
st.markdown("---")
st.header("📈 Accuracy by Prediction Window")
col1, col2 = st.columns(2)

with col1:
    fig_spearman = go.Figure()
    fig_spearman.add_trace(go.Scatter(
        x=results_df["Days"], y=results_df["Spearman ρ"],
        mode="lines+markers+text",
        text=results_df["Window"],
        textposition="top center",
        line=dict(color="royalblue", width=3),
        marker=dict(size=12, color="royalblue"),
        name="Spearman ρ",
    ))
    fig_spearman.update_layout(
        title="Spearman Correlation by Window",
        xaxis_title="Days Since Install",
        yaxis_title="Spearman ρ",
        height=400,
        xaxis=dict(tickvals=[1, 3, 5, 7], ticktext=["D1", "D3", "D5", "D7"]),
    )
    st.plotly_chart(fig_spearman, use_container_width=True)

with col2:
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Scatter(
        x=results_df["Days"], y=results_df["Lift@10%"],
        mode="lines+markers+text",
        text=results_df["Window"],
        textposition="top center",
        line=dict(color="#FF6600", width=3),
        marker=dict(size=12, color="#FF6600"),
        name="Lift@10%",
    ))
    fig_lift.update_layout(
        title="Lift@10% by Window",
        xaxis_title="Days Since Install",
        yaxis_title="Lift@10% (%)",
        height=400,
        xaxis=dict(tickvals=[1, 3, 5, 7], ticktext=["D1", "D3", "D5", "D7"]),
    )
    st.plotly_chart(fig_lift, use_container_width=True)

# ── Chart 2: Retention & Decay ─────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    fig_retention = go.Figure()
    fig_retention.add_trace(go.Bar(
        x=results_df["Window"],
        y=results_df["Spearman Retention %"],
        name="Spearman Retention",
        marker_color="royalblue",
        text=results_df["Spearman Retention %"].astype(str) + "%",
        textposition="outside",
    ))
    fig_retention.add_trace(go.Bar(
        x=results_df["Window"],
        y=results_df["Lift Retention %"],
        name="Lift Retention",
        marker_color="#FF6600",
        text=results_df["Lift Retention %"].astype(str) + "%",
        textposition="outside",
    ))
    fig_retention.add_hline(y=95, line_dash="dash", line_color="green",
                            annotation_text="95% threshold")
    fig_retention.update_layout(
        title="Accuracy Retention vs D7 Baseline (%)",
        barmode="group", height=400,
        yaxis=dict(range=[0, 115]),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_retention, use_container_width=True)

with col4:
    # Predicted vs actual scatter — use best available window (D3 preferred)
    scatter_window = "D3" if "D3" in preds else list(preds.keys())[-1]
    y_w_true, y_w_pred = preds[scatter_window]
    n_samp = min(3000, len(y_w_true))
    sample_idx = np.random.RandomState(42).choice(len(y_w_true), n_samp, replace=False)
    y_sample = y_w_true[sample_idx]
    pred_sample = y_w_pred[sample_idx]
    payers_mask = y_sample > 0

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=pred_sample[~payers_mask], y=y_sample[~payers_mask],
        mode="markers", name="Non-Payers",
        marker=dict(color="#e74c3c", size=3, opacity=0.4),
    ))
    fig_scatter.add_trace(go.Scatter(
        x=pred_sample[payers_mask], y=y_sample[payers_mask],
        mode="markers", name="Payers",
        marker=dict(color="#2ecc71", size=4, opacity=0.6),
    ))
    max_val = max(float(y_sample.max()), float(pred_sample.max()))
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect", line=dict(color="gray", dash="dash"),
    ))
    fig_scatter.update_layout(
        title=f"{scatter_window} Predicted vs Actual LTV30 (sample)",
        xaxis_title="Predicted LTV30",
        yaxis_title="Actual LTV30",
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Summary Table ──────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Performance Table")
tbl = results_df[["Window", "Spearman ρ", "Lift@10%", "RMSE",
                   "Spearman Retention %", "Lift Retention %"]].copy()
tbl["RMSE"] = tbl["RMSE"].apply(lambda v: format_currency(convert_vnd(v, cur["code"]), cur["code"]))
tbl.columns = ["Window", "Spearman ρ", "Lift@10% (%)", f"RMSE ({cur['symbol']})",
               "Spearman Retention %", "Lift Retention %"]
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Insights")
st.markdown(f"- **D3 is the practical sweet spot** — retains {d3_row['Spearman Retention %']:.1f}% of D7 Spearman accuracy, enabling predictions **4 days earlier** (based on actual D3 data)")
st.markdown(f"- **D1 retains {d1_row['Spearman Retention %']:.1f}%** of D7 accuracy — sufficient for binary triage (likely payer vs unlikely)")
st.markdown("- Diminishing returns after D5 — most predictive signal is captured by D3")
st.markdown("### 🎯 Recommended Deployment")
st.markdown("- **D1:** Auto-pause underperforming campaigns within 24 hours")
st.markdown("- **D3:** Primary scoring — seed generation and bid optimization")
st.markdown("- **D5:** Refinement — update predictions for borderline users")
st.markdown("- **D7:** Final scoring — complete picture for model evaluation")
