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
# Test dataset definitions
# ---------------------------------------------------------------------------
TEST_DATASETS = {
    "Test 1 — OOT Near (Jan 9–13)": {
        "file": "cfm_pltv_test1.csv",
        "rows": "118k",
        "dates": "2026-01-09 to 2026-01-13",
        "description": "Closer to training window — easier generalization test.",
    },
    "Test 2 — OOT Far (Jan 14–18)": {
        "file": "cfm_pltv_test2.csv",
        "rows": "82k",
        "dates": "2026-01-14 to 2026-01-18",
        "description": "Further from training window — harder generalization test.",
    },
}


# ---------------------------------------------------------------------------
# Sidebar (called from every page)
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the shared sidebar controls: training row limit, currency toggle, info."""
    st.sidebar.title("🎯 CFM Decision Intelligence")

    # Currency toggle at the top
    st.sidebar.markdown("### 💱 Currency Display")
    currency = st.sidebar.radio(
        "Show values in:",
        ["VND (₫)", "USD ($)"],
        index=0,
        horizontal=True,
        help=f"Toggle between Vietnamese Dong and US Dollar.  \n"
             f"Conversion rate: 1 USD ≈ ₫{VND_TO_USD_RATE:,.0f}",
    )
    st.session_state["currency"] = "VND" if "VND" in currency else "USD"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Training Data")

    train_path = DATA_DIR / "cfm_pltv_train.csv"
    if not train_path.exists():
        st.sidebar.warning("⚠️ No training data found")
        st.sidebar.info("📤 Use the **Data Upload** page to upload your dataset")
        st.session_state["data_missing"] = True
        return

    st.session_state["data_missing"] = False
    train_size = os.path.getsize(train_path) / 1e6
    st.sidebar.caption(f"Source: `cfm_pltv_train.csv` ({train_size:.0f} MB)")

    max_rows = st.sidebar.slider(
        "Training rows to load",
        min_value=10_000,
        max_value=1_730_000,
        value=50_000,
        step=10_000,
        help="Training data: **2025-12-16 to 2026-01-08** (1.73M rows).  \n"
             "Start small for fast iteration, increase for final evaluation.",
    )
    st.session_state["max_rows"] = max_rows
    st.session_state["dataset_choice"] = "real"

    st.sidebar.caption(f"📊 Loading up to **{max_rows:,}** training rows")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📑 Framework Layers")
    st.sidebar.markdown(
        "1. 📋 Decision Definition\n"
        "2. 🔧 Features & Model\n"
        "3. 📊 Evaluation & Insights\n"
        "4. 🎮 Action & Simulation\n"
        "5. 🔄 Feedback & Learning\n"
        "6. 🔬 Diagnostics"
    )


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading training data…")
def load_data(max_rows: int = 50_000) -> pd.DataFrame:
    """Load training data (2025-12-16 to 2026-01-08). Excludes all test sets."""
    csv_path = DATA_DIR / "cfm_pltv_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}. "
            f"Please upload your dataset using the Data Upload page."
        )
    return pd.read_csv(csv_path, nrows=max_rows, low_memory=False)


def get_data() -> pd.DataFrame:
    """Load training data using current session settings."""
    mr = st.session_state.get("max_rows", 50_000)
    return load_data(mr)


@st.cache_data(show_spinner="Loading test data…")
def load_test(filename: str) -> pd.DataFrame:
    """Load a test dataset by filename."""
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {csv_path}. "
            f"Run utils/split_oot.py first."
        )
    return pd.read_csv(csv_path, low_memory=False)


def load_test_1() -> pd.DataFrame:
    """Test 1 — OOT Near: 2026-01-09 to 2026-01-13 (118k rows)."""
    return load_test("cfm_pltv_test1.csv")


def load_test_2() -> pd.DataFrame:
    """Test 2 — OOT Far: 2026-01-14 to 2026-01-18 (82k rows)."""
    return load_test("cfm_pltv_test2.csv")


def get_test_data(choice: str) -> pd.DataFrame:
    """Load a test dataset by user choice key."""
    info = TEST_DATASETS[choice]
    return load_test(info["file"])


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
