"""
shared.py — Shared configuration, data loading, and model registry for the Streamlit app.
Imported by every page to provide consistent sidebar controls and data access.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Feature definitions — organized into logical groups for the Model Registry
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "💰 Payment (D7)": {
        "description": "Early monetization signals — strongest predictors of future LTV.",
        "features": {
            "rev_d7":   "Total revenue in first 7 days (₫)",
            "txn_cnt_d7": "Number of payment transactions in D0–D7",
            "first_charge_day_offset_d7": "Days from install to first purchase (null = non-payer → filled as -1)",
        },
    },
    "🎮 Gameplay (D7)": {
        "description": "In-game behavior — engagement depth and skill signals.",
        "features": {
            "games_d7":           "Total matches played in D0–D7",
            "win_rate_d7":        "Win rate across all matches",
            "avg_game_duration_d7": "Average match duration (seconds)",
            "avg_score_d7":       "Average in-game score",
            "kills_d7":           "Total kills",
            "deaths_d7":          "Total deaths",
            "assists_d7":         "Total assists",
            "kd_d7":              "Kill/Death ratio",
            "max_level_game_d7":  "Highest player level reached in matches",
            "max_ladderlevel_d7": "Highest ladder/rank level",
        },
    },
    "📱 Login Activity (D7)": {
        "description": "Session frequency and platform diversity — retention proxies.",
        "features": {
            "login_rows_d7":           "Total login events in D0–D7",
            "active_days_d7":          "Distinct days with a login (0–7)",
            "loginchannel_variety_d7":  "# distinct login channels used",
            "network_variety_d7":      "# distinct network types seen",
            "clientversion_variety_d7": "# distinct client versions seen",
            "max_level_seen_d7":       "Max account level observed in logins",
            "max_ladderscore_d7":      "Max ladder score observed in logins",
        },
    },
    "📣 UA Attribution": {
        "description": "Acquisition source — helps model generalize across campaigns.",
        "features": {
            "media_source":        "Ad network (facebook, google, tiktok, …)",
            "first_country_code":  "Country at install",
            "first_os":            "OS at install (android / ios)",
            "first_login_channel": "First login method (google_play, guest, …)",
        },
    },
}

# Flat lists for convenience
ALL_NUMERIC_FEATURES = []
ALL_CAT_FEATURES = []
for grp, info in FEATURE_GROUPS.items():
    for feat in info["features"]:
        if feat in ("media_source", "first_country_code", "first_os", "first_login_channel"):
            ALL_CAT_FEATURES.append(feat)
        else:
            ALL_NUMERIC_FEATURES.append(feat)

ALL_FEATURES = ALL_NUMERIC_FEATURES + ALL_CAT_FEATURES

# ---------------------------------------------------------------------------
# Baseline heuristics — simple single-column ranking strategies
# ---------------------------------------------------------------------------
BASELINE_HEURISTICS = {
    "rev_d7 (D7 Revenue)": {
        "column": "rev_d7",
        "color": "#e74c3c",
        "description": "Rank users by their first-week spend. The simplest and most common heuristic.",
    },
    "login_rows_d7 (Login Volume)": {
        "column": "login_rows_d7",
        "color": "#2ecc71",
        "description": "Rank by raw login event count — a proxy for engagement frequency.",
    },
    "active_days_d7 (Active Days)": {
        "column": "active_days_d7",
        "color": "#f39c12",
        "description": "Rank by number of distinct active days — retention strength signal.",
    },
    "games_d7 (Games Played)": {
        "column": "games_d7",
        "color": "#9b59b6",
        "description": "Rank by total matches played — gameplay engagement depth.",
    },
    "kd_d7 (Kill/Death Ratio)": {
        "column": "kd_d7",
        "color": "#1abc9c",
        "description": "Rank by K/D ratio — skill-based heuristic, tests if better players spend more.",
    },
}



# ---------------------------------------------------------------------------
# Top Menu (called from every page)
# ---------------------------------------------------------------------------
def render_top_menu():
    """Render horizontal top menu with title, currency, and framework navigation."""
    # Custom CSS: wider layout + hide default sidebar nav (we use custom page_link)
    st.markdown("""
        <style>
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 80% !important;
            width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Top bar — title only
    st.markdown("### 🔥 CrossFire Decision Intelligence")
    
# ---------------------------------------------------------------------------
# Shared report renderer — handles markdown with embedded local images
# ---------------------------------------------------------------------------
def render_report_md(report_path, expander_label: str = "📄 Full Report", expanded: bool = False):
    """
    Display a markdown report file inside an expander.
    Handles embedded image references (![alt](plots/foo.png)) by rendering
    them as st.image() calls so they display correctly in Streamlit.
    """
    import re
    if not Path(report_path).exists():
        return
    with st.expander(expander_label, expanded=expanded):
        report_text = Path(report_path).read_text(encoding="utf-8")
        parts = re.split(r'!\[([^\]]*)\]\(([^\)]+)\)', report_text)
        i = 0
        while i < len(parts):
            if i % 3 == 0:
                if parts[i].strip():
                    st.markdown(parts[i])
            elif i % 3 == 1:
                alt_text = parts[i]
                img_rel = parts[i + 1]
                img_path = Path(report_path).parent / img_rel
                if img_path.exists():
                    st.image(str(img_path), caption=alt_text, width=700)
                else:
                    st.caption(f"_(chart not found: {img_rel})_")
                i += 1  # skip the path part
            i += 1


# ---------------------------------------------------------------------------
# Sidebar (called from every page)
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the shared sidebar controls: custom navigation with emojis, training data config."""
    # Auto-load default model on first page visit (idempotent)
    from model_registry import auto_load_default_model
    auto_load_default_model()

    # Increase sidebar font size (+2px)
    st.markdown("""
        <style>
        .st-emotion-cache-1gwvy71 *,
        .stSidebar *,
        [data-testid="stSidebar"] *,
        [data-testid="stSidebarContent"] *,
        [data-testid="stSidebarUserContent"] * {
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Detect current page to determine if pLTV 30d Analysis should be expanded
    import inspect
    current_page = "unknown"
    for frame_info in inspect.stack():
        fname = Path(frame_info.filename).stem
        if fname and (fname[0].isdigit() or fname.startswith("Home")):
            current_page = fname
            break
    
    # Pages that belong to pLTV 30d Analysis section
    pltv_pages = {
        "1_Decision_Definition",
        "2_Features_and_Model", 
        "3_Evaluation_and_Insights",
        "4_Action_and_Simulation",
        "5_Feedback_and_Learning",
        "6_Summary"
    }
    is_pltv_page = current_page in pltv_pages

    # Pages that belong to AI-Generated Reports section
    ai_pages = {
        "3h_Whale_Segmentation",
        "3i_Time_to_First_Purchase",
        "3j_Channel_Whale_Quality",
        "3k_Churn_Prediction",
        "3l_Skill_Spend_Correlation",
    }
    is_ai_page = current_page in ai_pages
    
    # ── Section A: Key Functions ──────────────────────────────────────
    st.sidebar.markdown('<h3 style="font-size: 18px;">🔧 Key Functions</h3>', unsafe_allow_html=True)
    st.sidebar.page_link("pages/0_Data_Upload.py", label="📤 Data Upload")
    st.sidebar.page_link("pages/7_Notebooks.py", label="📓 Notebooks")
    
    # ── Section B: Analysis Reports ───────────────────────────────────
    st.sidebar.markdown('<h3 style="font-size: 18px;">📊 Analysis Reports</h3>', unsafe_allow_html=True)
    
    # pLTV 30d Analysis (expand if on a child page, collapse otherwise)
    with st.sidebar.expander("📈 pLTV 30d Analysis", expanded=is_pltv_page):
        st.page_link("pages/1_Decision_Definition.py", label="🎯 Definition")
        st.page_link("pages/2_Features_and_Model.py", label="  ⚔️ Features & Model")
        st.page_link("pages/3_Evaluation_and_Insights.py", label="  📊 Model Evaluation")
        st.page_link("pages/4_Action_and_Simulation.py", label="  🎮 Action & Simulation")
        st.page_link("pages/5_Feedback_and_Learning.py", label="  📉 Cohort Stability")
        st.page_link("pages/6_Summary.py", label="  📋 Summary & Next Steps")
    
    # Standalone reports
    st.sidebar.page_link("pages/3b_Late_Payer_Analysis.py", label="🔍 Late Payer Analysis")
    st.sidebar.page_link("pages/3c_Temporal_Analysis.py", label="📈 Temporal Analysis")
    st.sidebar.page_link("pages/3d_Cohort_Comparison.py", label="👥 Media Cohort Comparison")
    st.sidebar.page_link("pages/3e_Causal_Inference.py", label="🔬 Causal Inference")
    st.sidebar.page_link("pages/3f_Seed_Optimization.py", label="🌱 Seed Optimization")
    st.sidebar.page_link("pages/3g_Real_Time_Scoring.py", label="⚡ Real-Time Scoring")

    # ── Section C: AI-Generated Reports ──────────────────────────────
    st.sidebar.markdown('<h2 style="font-size: 18px;">🤖 Whales Focused Reports</h2>', unsafe_allow_html=True)
    with st.sidebar.expander("🐋 Whale Intelligence", expanded=is_ai_page):
        st.page_link("pages/3h_Whale_Segmentation.py", label="🐋 Whale Segmentation")
        st.page_link("pages/3i_Time_to_First_Purchase.py", label="⏱️ Time-to-First-Purchase")
        st.page_link("pages/3j_Channel_Whale_Quality.py", label="📡 Channel × Whale Quality")
        st.page_link("pages/3k_Churn_Prediction.py", label="📉 Churn Prediction")
        st.page_link("pages/3l_Skill_Spend_Correlation.py", label="🎯 Skill-to-Spend")

    # ── Currency selector ─────────────────────────────────────────
    st.sidebar.markdown("---")
    currency = st.sidebar.selectbox(
        "💱 Currency",
        ["🇻🇳 VND (₫)", "💵 USD ($)"],
        index=0 if st.session_state.get("currency", "VND") == "VND" else 1,
        help=f"VND ↔ USD (1 USD ≈ ₫{VND_TO_USD_RATE:,.0f})"
    )
    st.session_state["currency"] = "VND" if "VND" in currency else "USD"

    # ── Dataset Registry sidebar ──────────────────────────────────
    from dataset_registry import render_dataset_sidebar, migrate_existing_datasets
    # Auto-register any legacy CSV files on first run
    migrate_existing_datasets()
    # Determine current page ID from the call stack
    import inspect
    page_id = "unknown"
    for frame_info in inspect.stack():
        fname = Path(frame_info.filename).stem
        # Match known page patterns (e.g., "1_Decision_Definition", "3b_Late_Payer_Analysis")
        if fname and (fname[0].isdigit() or fname.startswith("Home")):
            page_id = fname
            break
    render_dataset_sidebar(page_id)


# ---------------------------------------------------------------------------
# Cached data loaders — all data flows through the Dataset Registry
# ---------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """Load data from the Dataset Registry (selected in sidebar).
    Auto-defaults to cfm_pltv_train if no dataset is bound."""
    from dataset_registry import load_dataset, list_datasets

    ds_id = st.session_state.get("current_dataset_id")

    # Auto-resolve: prefer cfm_pltv_train, then any registered train-role dataset
    if not ds_id:
        all_ds = list_datasets()
        # Priority 1: exact name match
        for candidate in ("cfm_pltv_train", "cfm_pltv"):
            if candidate in all_ds:
                ds_id = candidate
                break
        # Priority 2: role == 'train'
        if not ds_id:
            for did, meta in all_ds.items():
                if meta.get("role") == "train":
                    ds_id = did
                    break
        # Priority 3: first available
        if not ds_id and all_ds:
            ds_id = next(iter(all_ds))
        # Fallback: legacy file
        if not ds_id:
            legacy = DATA_DIR / "cfm_pltv_train.csv"
            if legacy.exists():
                st.session_state["current_dataset_id"] = "cfm_pltv_train"
                ds_id = "cfm_pltv_train"
        if not ds_id:
            st.error("No dataset found. Please upload a dataset on the **� Data Upload** page.")
            st.stop()
        st.session_state["current_dataset_id"] = ds_id

    mr = st.session_state.get("max_rows", 0)
    df = load_dataset(ds_id, max_rows=mr)
    st.session_state["actual_row_count"] = len(df)
    return df


@st.cache_data(show_spinner="Loading dataset…")
def load_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Load a CSV file with caching keyed by (path, mtime).
    Switching to a different dataset always triggers a fresh load + full recalculation.
    Switching back to a previously-loaded dataset is instant (served from cache).
    mtime is used purely as a cache-busting key — pass os.path.getmtime(path).
    """
    return pd.read_csv(path, low_memory=False)


def get_registry_path() -> tuple:
    """Return (csv_path_str, file_mtime) for the currently selected registry dataset.
    Used by pages that pass path+mtime to cached compute functions."""
    ds_id = st.session_state.get("current_dataset_id")
    if not ds_id:
        st.error("No dataset selected. Please select a dataset from the **📚 Dataset Registry** in the sidebar.")
        st.stop()

    from dataset_registry import list_datasets, DATA_DIR as DS_DATA_DIR
    datasets = list_datasets()
    if ds_id not in datasets:
        st.error(f"Dataset `{ds_id}` not found in registry. Please select a valid dataset.")
        st.stop()

    meta = datasets[ds_id]
    fpath = DS_DATA_DIR / meta["filename"]
    mtime = os.path.getmtime(fpath) if fpath.exists() else 0.0
    return str(fpath), mtime


def get_active_model():
    """Return (model, model_features) from session state.
    Checks 'model' (trained this session) first, then 'loaded_model' (loaded from registry).
    Returns (None, []) if no model is available."""
    if "model" in st.session_state:
        return st.session_state["model"], st.session_state.get("model_features", [])
    if "loaded_model" in st.session_state:
        meta = st.session_state.get("loaded_model_metadata", {})
        return st.session_state["loaded_model"], meta.get("features", [])
    return None, []


# ---------------------------------------------------------------------------
# Currency formatting with USD/VND toggle
# ---------------------------------------------------------------------------
# Current VND to USD conversion rate (approximate, update as needed)
VND_TO_USD_RATE = 24000.0  # 1 USD ≈ 24,000 VND

def format_currency(amount: float, currency: str = "VND", decimals: int = None) -> str:
    """
    Format amount in VND or USD based on user preference.
    
    Args:
        amount: Amount in VND (base currency)
        currency: "VND" or "USD"
        decimals: Number of decimal places (auto-set if None)
    """
    if pd.isna(amount):
        return "₫0" if currency == "VND" else "$0"
    
    if currency == "USD":
        usd_amount = amount / VND_TO_USD_RATE
        dec = 2 if decimals is None else decimals
        return f"${usd_amount:,.{dec}f}"
    else:  # VND
        dec = 0 if decimals is None else decimals
        return f"₫{amount:,.{dec}f}"

def format_vnd(amount: float, decimals: int = 0) -> str:
    """Legacy function - format as VND. Use format_currency() for toggle support."""
    return format_currency(amount, "VND", decimals)


def convert_vnd(values, currency: str = "VND"):
    """Convert VND values to USD if currency is USD. Works with scalars, arrays, Series."""
    if currency == "USD":
        return values / VND_TO_USD_RATE
    return values


def get_currency_info() -> dict:
    """Return current currency settings from session state."""
    currency = st.session_state.get("currency", "VND")
    return {
        "code": currency,
        "symbol": "₫" if currency == "VND" else "$",
        "label": "VND" if currency == "VND" else "USD",
    }



# ---------------------------------------------------------------------------
# Baseline ranker
# ---------------------------------------------------------------------------
def compute_baseline_ranking(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return predicted scores from a simple heuristic: rank by a single column."""
    return df[column].fillna(0).values.astype(float)


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------
def get_selected_features() -> dict:
    """Return the currently selected features from session state, grouped."""
    if "registry_features" not in st.session_state:
        # Default: all features selected
        st.session_state["registry_features"] = {
            grp: list(info["features"].keys())
            for grp, info in FEATURE_GROUPS.items()
        }
    return st.session_state["registry_features"]


def get_flat_selected_features() -> tuple:
    """Return (numeric_list, cat_list) of currently selected features."""
    sel = get_selected_features()
    num, cat = [], []
    for grp, feats in sel.items():
        for f in feats:
            if f in ALL_CAT_FEATURES:
                cat.append(f)
            else:
                num.append(f)
    return num, cat


# ---------------------------------------------------------------------------
# Dataset role helpers
# ---------------------------------------------------------------------------
def list_datasets_by_role() -> dict:
    """
    Return a dict mapping role -> (dataset_name, path, size_mb, mtime).
    Roles: 'train', 'test', 'recent'.
    Falls back to filename-based detection if registry metadata is missing.
    Excludes D135 part files.
    """
    roles: dict = {"train": None, "test": None, "recent": None}

    # First pass: registry metadata
    try:
        from dataset_registry import list_datasets
        for ds_id, meta in list_datasets().items():
            role = meta.get("role")
            fname = meta.get("filename", "")
            path = DATA_DIR / fname
            if not path.exists():
                continue
            if role in roles:
                roles[role] = {
                    "name": meta["name"],
                    "path": str(path),
                    "size_mb": meta.get("size_mb", path.stat().st_size / 1e6),
                    "mtime": path.stat().st_mtime,
                    "split_info": meta.get("split_info", ""),
                }
    except Exception:
        pass

    # Second pass: filename-based fallback for any role still missing
    fallback_map = {
        "train":  ["cfm_pltv_train.csv"],
        "test":   ["cfm_pltv_test.csv", "cfm_pltv_test1.csv"],
        "recent": ["cfm_pltv_recent.csv"],
    }
    for role, candidates in fallback_map.items():
        if roles[role] is None:
            for fname in candidates:
                p = DATA_DIR / fname
                if p.exists():
                    roles[role] = {
                        "name": p.stem,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / 1e6, 1),
                        "mtime": p.stat().st_mtime,
                        "split_info": "",
                    }
                    break

    return roles


def render_dataset_role_selector(
    available_roles: list,
    default_role: str = "train",
    key_prefix: str = "ds_role",
) -> tuple:
    """
    Render a Train / Test / Recent tab selector for pages that support multiple dataset roles.
    Returns (selected_role, dataset_info_dict_or_None).

    available_roles: subset of ['train', 'test', 'recent'] to show.
    """
    role_meta = list_datasets_by_role()

    role_labels = {
        "train":  "🏋️ Train (in-sample)",
        "test":   "🧪 Test (holdout)",
        "recent": "🆕 Recent (live scoring)",
    }
    role_descriptions = {
        "train":  "Mature users (≥30d), 80% split — used for model training and all analysis pages.",
        "test":   "Mature users (≥30d), 20% holdout — unbiased evaluation of model performance.",
        "recent": "Users installed <30d ago — LTV30 not yet realized. Score with trained model only.",
    }

    tabs = st.tabs([role_labels[r] for r in available_roles])
    selected_role = st.session_state.get(f"{key_prefix}_role", default_role)

    for i, (tab, role) in enumerate(zip(tabs, available_roles)):
        with tab:
            st.caption(role_descriptions[role])
            info = role_meta.get(role)
            if info is None:
                st.warning(
                    f"No **{role}** dataset found. "
                    "Upload a new dataset on the **📤 Data Upload** page to generate it automatically."
                )
            else:
                st.caption(
                    f"**{info['name']}** — {info['size_mb']:.1f} MB"
                    + (f" | {info['split_info']}" if info["split_info"] else "")
                )
            # Track which tab is active via a hidden selectbox trick
            if st.session_state.get(f"{key_prefix}_active_tab") == i:
                selected_role = role
                st.session_state[f"{key_prefix}_role"] = role

    # Streamlit tabs don't expose active index natively — use a selectbox as fallback
    chosen_role = st.selectbox(
        "Dataset role",
        available_roles,
        index=available_roles.index(default_role) if default_role in available_roles else 0,
        format_func=lambda r: role_labels[r],
        key=f"{key_prefix}_select",
        label_visibility="collapsed",
    )
    info = role_meta.get(chosen_role)
    return chosen_role, info
