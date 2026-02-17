"""
Page 2 — Features & Model + Model Registry
Feature profiling, correlations, interactive feature selection, and model training.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar, get_data, get_test_data, format_currency, convert_vnd,
    get_currency_info, REPORTS_DIR,
    FEATURE_GROUPS, ALL_NUMERIC_FEATURES, ALL_CAT_FEATURES, TEST_DATASETS,
    get_selected_features, get_flat_selected_features,
)
from model_registry import save_model, load_model, show_model_management_ui

render_sidebar()

st.title("🔧 Layer 2 — Features, Model Registry & Training")

if st.session_state.get("data_missing", False):
    st.warning("⚠️ No training data found")
    st.info("Please upload your dataset using the **📤 Data Upload** page in the sidebar.")
    st.stop()

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows (2025-12-16 to 2026-01-08)")
st.markdown("---")

# Report
report_path = REPORTS_DIR / "feature_store_overview.md"
if report_path.exists():
    with st.expander("📄 Feature Store Report", expanded=False):
        st.markdown(report_path.read_text(encoding="utf-8"))

# =====================================================================
# SECTION 1 — Feature Profiling
# =====================================================================
st.header("📊 Feature Profiling")
cur = get_currency_info()
available_numeric = [f for f in ALL_NUMERIC_FEATURES if f in df.columns]

# Convert monetary columns for the profiling table
monetary_cols = {"rev_d7", "first_charge_day_offset_d7"}  # rev_d7 is monetary
profile_df = df[available_numeric].copy()
for c in available_numeric:
    if c == "rev_d7":
        profile_df[c] = convert_vnd(profile_df[c], cur["code"])

profile = profile_df.describe().T
profile["null_pct"] = (df[available_numeric].isnull().sum() / len(df) * 100).round(1)
st.dataframe(profile[["null_pct", "mean", "std", "min", "50%", "max"]].rename(
    columns={"null_pct": "Null %", "50%": "Median"}
), width='stretch')
if cur["code"] == "USD":
    st.caption("💱 Monetary values (rev_d7) shown in USD")

# --- Correlation with LTV30 ---
st.header("Feature Correlations with LTV30")
st.markdown("> **Why this matters:** Features with high Spearman ρ are strong ranking signals. "
            "A model that combines them should outperform any single-feature heuristic.")

currency_label = f"LTV30 ({cur['symbol']})"

corr_with_ltv = df[available_numeric + ["ltv30"]].corr(method="spearman")["ltv30"].drop("ltv30").sort_values(ascending=False)
fig_corr = px.bar(
    x=corr_with_ltv.values, y=corr_with_ltv.index,
    orientation="h", title=f"Spearman Correlation with {currency_label}",
    labels={"x": "Spearman ρ", "y": "Feature"},
    color=corr_with_ltv.values, color_continuous_scale="RdYlGn",
)
fig_corr.update_layout(height=500, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_corr, width='stretch')

# --- Cohort Distributions ---
with st.expander("📈 Cohort Distributions", expanded=False):
    tab1, tab2, tab3 = st.tabs(["By Media Source", "By Country", "By OS"])
    with tab1:
        fig = px.histogram(df, x="media_source", color="is_payer_30",
                           barmode="group", title="User Count by Media Source (Payer vs Non-Payer)")
        st.plotly_chart(fig, width='stretch')
    with tab2:
        fig = px.histogram(df, x="first_country_code", color="is_payer_30",
                           barmode="group", title="User Count by Country")
        st.plotly_chart(fig, width='stretch')
    with tab3:
        fig = px.histogram(df, x="first_os", color="is_payer_30",
                           barmode="group", title="User Count by OS")
        st.plotly_chart(fig, width='stretch')

# =====================================================================
# SECTION 2 — Model Registry (Feature Selection)
# =====================================================================
st.markdown("---")
st.header("🗂️ Model Registry — Choose Your Features")
st.markdown(
    "Select which feature groups and individual features to include in training.  \n"
    "**Tip:** Start with Payment features only (the strongest signal), then gradually "
    "add other groups to see how evaluation metrics change."
)

# Initialize: set every individual feature checkbox key to True on first load
for grp, info in FEATURE_GROUPS.items():
    for feat in info["features"]:
        if f"feat_{feat}" not in st.session_state:
            st.session_state[f"feat_{feat}"] = True

# Callback factory: when "Select all" toggles, push that value into every
# individual feature checkbox key in the group so Streamlit picks it up.
def _on_group_toggle(grp_key, feat_keys):
    val = st.session_state[grp_key]
    for fk in feat_keys:
        st.session_state[fk] = val

selected_features = {}

for grp, info in FEATURE_GROUPS.items():
    with st.expander(f"{grp}  —  {info['description']}", expanded=True):
        all_feats = list(info["features"].keys())
        feat_keys = [f"feat_{f}" for f in all_feats]

        # Derive "all selected?" from individual checkbox states
        all_selected = all(st.session_state.get(fk, True) for fk in feat_keys)

        col_toggle, col_info = st.columns([1, 3])
        with col_toggle:
            grp_toggle_key = f"grp_toggle_{grp}"
            st.checkbox(
                "Select all", value=all_selected,
                key=grp_toggle_key,
                on_change=_on_group_toggle,
                args=(grp_toggle_key, feat_keys),
            )
        with col_info:
            st.caption(f"{len(all_feats)} features available")

        # Individual feature checkboxes
        cols = st.columns(min(3, len(all_feats)))
        grp_selected = []
        for i, (feat, desc) in enumerate(info["features"].items()):
            with cols[i % len(cols)]:
                checked = st.checkbox(
                    f"`{feat}`",
                    key=f"feat_{feat}",
                    help=desc,
                )
                if checked:
                    grp_selected.append(feat)

        selected_features[grp] = grp_selected

# Persist to session
st.session_state["registry_features"] = selected_features

# Summary
num_sel, cat_sel = get_flat_selected_features()
total_sel = len(num_sel) + len(cat_sel)
st.markdown(f"**Selected:** {total_sel} features "
            f"({len(num_sel)} numeric, {len(cat_sel)} categorical)")

if total_sel == 0:
    st.warning("⚠️ No features selected. Select at least one feature to train a model.")

# =====================================================================
# SECTION 3 — Model Training
# =====================================================================
st.markdown("---")

# Model management UI
show_model_management_ui()
# Show loaded model info
if "loaded_model" in st.session_state:
    loaded_name = st.session_state.get("loaded_model_name", "Unknown")
    loaded_metadata = st.session_state.get("loaded_model_metadata", {})
    
    st.success(f"✅ **Loaded from disk:** {loaded_name}")
    
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.metric("Features", loaded_metadata.get('n_features', 'N/A'))
    with col_meta2:
        st.metric("Dataset Size", f"{loaded_metadata.get('dataset_size', 0):,}")
    with col_meta3:
        st.metric("Saved", loaded_metadata.get('training_date', 'N/A'))
    
    if st.button("🔄 Use This Model", type="primary"):
        st.session_state["model"] = st.session_state["loaded_model"]
        st.session_state["model_features"] = loaded_metadata.get('features', [])
        st.success("✅ Model activated in session!")
        st.rerun()

elif "model" in st.session_state:
    feat_list = st.session_state.get("model_features", [])
    st.info(f"📦 **Active model** in session — trained with {len(feat_list)} features: "
            f"`{'`, `'.join(feat_list[:8])}`{'…' if len(feat_list) > 8 else ''}")
else:
    st.info("No model in session. Train a model above or load one from disk below.")

st.header("🚀 Train Model")

st.markdown(
    "Click **Train** to fit an XGBoost model with your selected features.  \n"
    "The trained model and baselines will be available on the **Evaluation** and **Action** pages."
)

report_train = REPORTS_DIR / "model_training.md"
if report_train.exists():
    with st.expander("📄 Model Training Report (reference)", expanded=False):
        st.markdown(report_train.read_text(encoding="utf-8"))

train_clicked = st.button("🚀 Train XGBoost pLTV Model", disabled=(total_sel == 0), type="primary")

if train_clicked and total_sel > 0:
    with st.spinner(f"Training on {len(df):,} rows with {total_sel} features…"):
        try:
            import xgboost as xgb

            # Prepare training data (all of df)
            feature_df_train = df[num_sel + cat_sel].copy()
            for c in cat_sel:
                le = LabelEncoder()
                feature_df_train[c] = le.fit_transform(feature_df_train[c].astype(str))
            if "first_charge_day_offset_d7" in feature_df_train.columns:
                feature_df_train["first_charge_day_offset_d7"] = feature_df_train["first_charge_day_offset_d7"].fillna(-1)

            X_train = feature_df_train
            y_train = np.log1p(df["ltv30"].clip(lower=0))

            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                objective="reg:squaredlogerror", random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)

            # Feature importance
            importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
            fig_imp = px.bar(
                x=importance.values, y=importance.index, orientation="h",
                title="Feature Importances (XGBoost — your selected features)",
                labels={"x": "Importance", "y": "Feature"},
                color=importance.values, color_continuous_scale="Tealgrn",
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Store in session for other pages
            st.session_state["model"] = model
            st.session_state["X_all"] = X_train
            st.session_state["y_all"] = y_train
            st.session_state["model_features"] = num_sel + cat_sel
            st.session_state["df_for_model"] = df

            st.success(f"✅ Model trained with **{total_sel} features** on **{len(X_train):,}** rows.")
            
            # Store training metadata for saving later
            st.session_state["last_training_metadata"] = {
                'model_type': 'XGBoost Regressor',
                'dataset_size': len(X_train),
                'n_features': total_sel,
                'features': num_sel + cat_sel,
                'numeric_features': num_sel,
                'categorical_features': cat_sel,
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'objective': 'reg:squaredlogerror'
                },
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'training_time': datetime.now().strftime('%H:%M:%S'),
            }

        except Exception as e:
            st.error(f"Training failed: {e}")

# Save model section (if model exists in session)
if "model" in st.session_state:
    st.markdown("### 💾 Save Current Model")
    
    col_save1, col_save2 = st.columns([2, 1])
    with col_save1:
        model_name = st.text_input(
            "Model name:",
            value=f"pltv_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="save_model_name"
        )
    
    with col_save2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save to Disk", type="primary", width='stretch'):
            model = st.session_state["model"]
            metadata = st.session_state.get("last_training_metadata", {})
            
            # Ensure metadata has required fields
            if not metadata:
                metadata = {
                    'model_type': 'XGBoost Regressor',
                    'dataset_size': len(st.session_state.get("X_all", [])),
                    'n_features': len(st.session_state.get("model_features", [])),
                    'features': st.session_state.get("model_features", []),
                    'training_date': datetime.now().strftime('%Y-%m-%d'),
                    'training_time': datetime.now().strftime('%H:%M:%S'),
                }
            
            with st.spinner("💾 Saving model..."):
                success, message = save_model(model, model_name, metadata)
            
            if success:
                st.toast(f"✅ {message}", icon="✅")
                st.success(f"✅ Model saved as: **{model_name}**")
                st.balloons()
            else:
                st.error(f"❌ Failed to save model")
                with st.expander("Error details"):
                    st.code(message)
    
    st.markdown("---")


