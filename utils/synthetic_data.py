"""
synthetic_data.py — Generate a realistic synthetic CFM pLTV dataset
matching the real schema for end-to-end demo before connecting the 350 MB file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_USERS = 30_000  # small enough for fast demo

COLUMNS_SCHEMA = {
    # UA attribution
    "vopenid": str,
    "roleid": str,
    "install_date": str,
    "game_id": str,
    "media_source": str,
    "campaign_id": str,
    "adset_id": str,
    "ad_id": str,
    "site_id": str,
    "first_os": str,
    "last_os": str,
    "first_country_code": str,
    "last_country_code": str,
    "first_login_channel": str,
    "last_login_channel": str,
    # Login D7
    "login_rows_d7": int,
    "active_days_d7": int,
    "loginchannel_variety_d7": int,
    "network_variety_d7": int,
    "clientversion_variety_d7": int,
    "max_level_seen_d7": int,
    "max_ladderscore_d7": float,
    # Gameplay D7
    "games_d7": int,
    "win_rate_d7": float,
    "avg_game_duration_d7": float,
    "avg_score_d7": float,
    "kills_d7": float,
    "deaths_d7": float,
    "assists_d7": float,
    "kd_d7": float,
    "max_level_game_d7": int,
    "max_ladderlevel_d7": float,
    # Payment
    "rev_d7": float,
    "txn_cnt_d7": int,
    "first_charge_day_offset_d7": float,
    # Labels
    "ltv30": float,
    "is_payer_30": int,
}

MEDIA_SOURCES = ["facebook", "google", "tiktok", "unity", "organic", "applovin", "ironsource"]
COUNTRIES = ["VN", "TH", "ID", "PH", "MY", "SG"]
OS_TYPES = ["android", "ios"]
LOGIN_CHANNELS = ["google_play", "app_store", "guest", "facebook"]


def generate(n: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # IDs
    vopenids = [f"vo_{i:07d}" for i in range(n)]
    roleids = [f"role_{i:07d}" for i in range(n)]
    install_dates = pd.date_range("2025-12-16", periods=45, freq="D")
    
    # UA features
    media = rng.choice(MEDIA_SOURCES, n, p=[0.25, 0.20, 0.15, 0.10, 0.15, 0.08, 0.07])
    country = rng.choice(COUNTRIES, n, p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05])
    os_first = rng.choice(OS_TYPES, n, p=[0.75, 0.25])
    login_ch = rng.choice(LOGIN_CHANNELS, n, p=[0.40, 0.20, 0.25, 0.15])
    
    # Login features
    active_days = rng.integers(0, 8, n)
    login_rows = active_days * rng.integers(1, 6, n)
    max_level = rng.integers(1, 30, n)
    max_ladder = rng.uniform(0, 2000, n)
    
    # Gameplay features
    games = rng.integers(0, 200, n)
    win_rate = rng.beta(2, 3, n)
    avg_duration = rng.lognormal(5.5, 0.5, n)  # ~250s avg
    avg_score = rng.lognormal(6, 1, n)
    kills = rng.poisson(15, n).astype(float)
    deaths = rng.poisson(12, n).astype(float)
    assists = rng.poisson(8, n).astype(float)
    kd = np.where(deaths > 0, kills / deaths, 0)
    
    # Payment — heavy right skew (most users free)
    is_payer_7 = rng.binomial(1, 0.08, n)
    rev_d7 = np.where(is_payer_7, rng.lognormal(1.5, 1.5, n), 0).round(2)
    txn_cnt = np.where(is_payer_7, rng.integers(1, 5, n), 0)
    first_charge_offset = np.where(is_payer_7, rng.integers(0, 8, n).astype(float), np.nan)
    
    # Label: LTV30 correlated with rev_d7, activity, engagement
    engagement_score = (
        0.3 * (active_days / 7)
        + 0.2 * (win_rate)
        + 0.15 * np.clip(games / 100, 0, 1)
        + 0.35 * np.clip(rev_d7 / 50, 0, 1)
    )
    ltv30_base = rev_d7 * rng.uniform(1.5, 4.0, n) + engagement_score * rng.uniform(0, 10, n)
    ltv30 = np.maximum(ltv30_base, 0).round(2)
    is_payer_30 = (ltv30 > 0).astype(int)
    
    df = pd.DataFrame({
        "vopenid": vopenids,
        "roleid": roleids,
        "install_date": rng.choice(install_dates, n).astype(str),
        "game_id": "cfm_vn",
        "media_source": media,
        "campaign_id": [f"camp_{rng.integers(100, 999)}" for _ in range(n)],
        "adset_id": [f"adset_{rng.integers(100, 999)}" for _ in range(n)],
        "ad_id": [f"ad_{rng.integers(1000, 9999)}" for _ in range(n)],
        "site_id": [f"site_{rng.integers(10, 99)}" for _ in range(n)],
        "first_os": os_first,
        "last_os": os_first,
        "first_country_code": country,
        "last_country_code": country,
        "first_login_channel": login_ch,
        "last_login_channel": login_ch,
        "login_rows_d7": login_rows,
        "active_days_d7": active_days,
        "loginchannel_variety_d7": rng.integers(1, 4, n),
        "network_variety_d7": rng.integers(1, 3, n),
        "clientversion_variety_d7": rng.integers(1, 3, n),
        "max_level_seen_d7": max_level,
        "max_ladderscore_d7": max_ladder.round(1),
        "games_d7": games,
        "win_rate_d7": win_rate.round(4),
        "avg_game_duration_d7": avg_duration.round(1),
        "avg_score_d7": avg_score.round(1),
        "kills_d7": kills,
        "deaths_d7": deaths,
        "assists_d7": assists,
        "kd_d7": kd.round(3),
        "max_level_game_d7": rng.integers(1, 30, n),
        "max_ladderlevel_d7": rng.uniform(0, 15, n).round(1),
        "rev_d7": rev_d7,
        "txn_cnt_d7": txn_cnt,
        "first_charge_day_offset_d7": first_charge_offset,
        "ltv30": ltv30,
        "is_payer_30": is_payer_30,
    })
    return df


def save_sample(output_path: str = None):
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent.parent / "data" / "cfm_pltv_sample.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic data saved: {output_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    save_sample()
