import pandas as pd
import numpy as np
import json
import sys

# Read CSV
print("Loading CSV...", file=sys.stderr)
df = pd.read_csv('/Users/lap16299/Documents/code/cfm-pltv/data/cfm_pltv_2025_12_16.csv', 
                  quotechar='"', low_memory=False)

results = {}

# ── 1. Basic shape ──
results['shape'] = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
results['columns'] = list(df.columns)
results['dtypes'] = {c: str(df[c].dtype) for c in df.columns}

# ── 2. Date range ──
df['install_date'] = pd.to_datetime(df['install_date'], errors='coerce')
results['date_range'] = {
    'min': str(df['install_date'].min()),
    'max': str(df['install_date'].max()),
    'unique_dates': int(df['install_date'].nunique()),
    'rows_per_date': df['install_date'].value_counts().sort_index().to_dict()
}
# Convert keys to string for JSON
results['date_range']['rows_per_date'] = {str(k): int(v) for k, v in results['date_range']['rows_per_date'].items()}

# ── 3. Nulls ──
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
results['null_pct'] = {c: float(v) for c, v in null_pct.items() if v > 0}

# ── 4. Numeric stats for key columns ──
numeric_cols = ['rev_d7', 'txn_cnt_d7', 'ltv30', 'login_rows_d7', 'active_days_d7',
                'games_d7', 'win_rate_d7', 'kills_d7', 'deaths_d7', 'kd_d7',
                'max_level_seen_d7', 'max_ladderscore_d7', 'avg_game_duration_d7',
                'avg_score_d7', 'first_charge_day_offset_d7']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

stats = df[numeric_cols].describe(percentiles=[.25, .5, .75, .9, .95, .99]).round(4)
results['numeric_stats'] = stats.to_dict()

# ── 5. LTV30 distribution (target variable) ──
ltv = df['ltv30'].dropna()
results['ltv30_distribution'] = {
    'mean': float(ltv.mean()),
    'median': float(ltv.median()),
    'std': float(ltv.std()),
    'min': float(ltv.min()),
    'max': float(ltv.max()),
    'pct_zero': float((ltv == 0).sum() / len(ltv) * 100),
    'pct_positive': float((ltv > 0).sum() / len(ltv) * 100),
    'total_payers': int((ltv > 0).sum()),
    'total_non_payers': int((ltv == 0).sum()),
}

# ── 6. Whale analysis ──
payers = df[df['ltv30'] > 0].copy()
results['whale_analysis'] = {
    'total_payers': int(len(payers)),
    'payer_rate': float(len(payers) / len(df) * 100),
}

if len(payers) > 0:
    # Define whale tiers by LTV30 percentiles among payers
    payer_ltv = payers['ltv30']
    thresholds = {
        'minnow_max': float(payer_ltv.quantile(0.5)),
        'dolphin_max': float(payer_ltv.quantile(0.9)),
        'whale_max': float(payer_ltv.quantile(0.99)),
    }
    results['whale_analysis']['thresholds'] = thresholds

    payers['segment'] = pd.cut(
        payers['ltv30'],
        bins=[-0.01, thresholds['minnow_max'], thresholds['dolphin_max'], thresholds['whale_max'], float('inf')],
        labels=['Minnow (<=p50)', 'Dolphin (p50-p90)', 'Whale (p90-p99)', 'Super Whale (>p99)']
    )
    seg = payers.groupby('segment', observed=True).agg(
        count=('ltv30', 'size'),
        total_rev=('ltv30', 'sum'),
        avg_ltv=('ltv30', 'mean'),
        median_ltv=('ltv30', 'median'),
        min_ltv=('ltv30', 'min'),
        max_ltv=('ltv30', 'max'),
    )
    seg['pct_of_payers'] = (seg['count'] / seg['count'].sum() * 100).round(2)
    seg['pct_of_revenue'] = (seg['total_rev'] / seg['total_rev'].sum() * 100).round(2)
    results['whale_analysis']['segments'] = seg.reset_index().to_dict(orient='records')
    for rec in results['whale_analysis']['segments']:
        rec['segment'] = str(rec['segment'])
        for k in rec:
            if isinstance(rec[k], (np.integer, np.int64)):
                rec[k] = int(rec[k])
            elif isinstance(rec[k], (np.floating, np.float64)):
                rec[k] = float(rec[k])

    # LTV30 percentile breakdown for payers
    pctiles = [10, 25, 50, 75, 90, 95, 99]
    results['whale_analysis']['payer_ltv_percentiles'] = {
        f'p{p}': float(payer_ltv.quantile(p/100)) for p in pctiles
    }

# ── 7. Correlation: payer at 7d vs ltv at 30d ──
df['is_payer_7d'] = (df['rev_d7'] > 0).astype(int)
df['is_payer_30'] = pd.to_numeric(df['is_payer_30'], errors='coerce')

# Confusion matrix: 7d payer vs 30d payer
ct = pd.crosstab(df['is_payer_7d'], df['is_payer_30'].fillna(0).astype(int))
results['payer_7d_vs_30d_crosstab'] = ct.to_dict()

# Among 7d payers: what % become 30d payers?
payers_7d = df[df['is_payer_7d'] == 1]
non_payers_7d = df[df['is_payer_7d'] == 0]

results['payer_7d_vs_ltv30'] = {
    'total_7d_payers': int(len(payers_7d)),
    'total_7d_non_payers': int(len(non_payers_7d)),
    '7d_payers_avg_ltv30': float(payers_7d['ltv30'].mean()) if len(payers_7d) > 0 else 0,
    '7d_payers_median_ltv30': float(payers_7d['ltv30'].median()) if len(payers_7d) > 0 else 0,
    '7d_non_payers_avg_ltv30': float(non_payers_7d['ltv30'].mean()) if len(non_payers_7d) > 0 else 0,
    '7d_non_payers_median_ltv30': float(non_payers_7d['ltv30'].median()) if len(non_payers_7d) > 0 else 0,
    '7d_payers_who_are_30d_payers_pct': float(
        (payers_7d['is_payer_30'] == 1).sum() / len(payers_7d) * 100
    ) if len(payers_7d) > 0 else 0,
    '7d_non_payers_who_are_30d_payers_pct': float(
        (non_payers_7d['is_payer_30'] == 1).sum() / len(non_payers_7d) * 100
    ) if len(non_payers_7d) > 0 else 0,
}

# Correlation of rev_d7 with ltv30
corr_rev7_ltv30 = df[['rev_d7', 'ltv30']].dropna().corr().iloc[0, 1]
results['payer_7d_vs_ltv30']['corr_rev_d7_ltv30'] = float(corr_rev7_ltv30)

# Among 7d payers, correlation of rev_d7 with ltv30
if len(payers_7d) > 0:
    corr_among_payers = payers_7d[['rev_d7', 'ltv30']].dropna().corr().iloc[0, 1]
    results['payer_7d_vs_ltv30']['corr_rev_d7_ltv30_among_7d_payers'] = float(corr_among_payers)

# ── 8. Feature correlations with LTV30 ──
feature_cols = ['login_rows_d7', 'active_days_d7', 'loginchannel_variety_d7',
                'network_variety_d7', 'clientversion_variety_d7', 'max_level_seen_d7',
                'max_ladderscore_d7', 'games_d7', 'win_rate_d7', 'avg_game_duration_d7',
                'avg_score_d7', 'kills_d7', 'deaths_d7', 'assists_d7', 'kd_d7',
                'max_level_game_d7', 'max_ladderlevel_d7', 'rev_d7', 'txn_cnt_d7',
                'first_charge_day_offset_d7']
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

corrs = df[feature_cols + ['ltv30']].corr()['ltv30'].drop('ltv30').sort_values(ascending=False)
results['feature_correlations_with_ltv30'] = {c: float(v) for c, v in corrs.items()}

# ── 9. Media source breakdown ──
ms = df.groupby('media_source').agg(
    count=('ltv30', 'size'),
    payer_rate=('is_payer_30', 'mean'),
    avg_ltv30=('ltv30', 'mean'),
    median_ltv30=('ltv30', 'median'),
    total_rev=('ltv30', 'sum'),
).sort_values('count', ascending=False).head(15)
ms['payer_rate'] = (ms['payer_rate'] * 100).round(2)
results['media_source_breakdown'] = ms.reset_index().to_dict(orient='records')
for rec in results['media_source_breakdown']:
    for k in rec:
        if isinstance(rec[k], (np.integer, np.int64)):
            rec[k] = int(rec[k])
        elif isinstance(rec[k], (np.floating, np.float64)):
            rec[k] = float(rec[k])

# ── 10. OS breakdown ──
os_stats = df.groupby('first_os').agg(
    count=('ltv30', 'size'),
    payer_rate=('is_payer_30', 'mean'),
    avg_ltv30=('ltv30', 'mean'),
    median_ltv30=('ltv30', 'median'),
).sort_values('count', ascending=False)
os_stats['payer_rate'] = (os_stats['payer_rate'] * 100).round(2)
results['os_breakdown'] = os_stats.reset_index().to_dict(orient='records')
for rec in results['os_breakdown']:
    for k in rec:
        if isinstance(rec[k], (np.integer, np.int64)):
            rec[k] = int(rec[k])
        elif isinstance(rec[k], (np.floating, np.float64)):
            rec[k] = float(rec[k])

# ── 11. Country breakdown ──
cc = df.groupby('first_country_code').agg(
    count=('ltv30', 'size'),
    payer_rate=('is_payer_30', 'mean'),
    avg_ltv30=('ltv30', 'mean'),
).sort_values('count', ascending=False).head(10)
cc['payer_rate'] = (cc['payer_rate'] * 100).round(2)
results['country_breakdown'] = cc.reset_index().to_dict(orient='records')
for rec in results['country_breakdown']:
    for k in rec:
        if isinstance(rec[k], (np.integer, np.int64)):
            rec[k] = int(rec[k])
        elif isinstance(rec[k], (np.floating, np.float64)):
            rec[k] = float(rec[k])

# ── 12. rev_d7 distribution among 7d payers ──
if len(payers_7d) > 0:
    rev7 = payers_7d['rev_d7']
    pctiles = [10, 25, 50, 75, 90, 95, 99]
    results['rev_d7_payer_distribution'] = {
        'count': int(len(rev7)),
        'mean': float(rev7.mean()),
        'median': float(rev7.median()),
        'std': float(rev7.std()),
        'percentiles': {f'p{p}': float(rev7.quantile(p/100)) for p in pctiles},
    }

# ── 13. Engagement features by payer status ──
eng_cols = ['active_days_d7', 'games_d7', 'max_level_seen_d7', 'avg_game_duration_d7', 'kills_d7']
eng_payer = df[df['is_payer_30'] == 1][eng_cols].mean()
eng_non_payer = df[df['is_payer_30'] == 0][eng_cols].mean()
results['engagement_by_payer_status'] = {
    'payer_means': {c: float(v) for c, v in eng_payer.items()},
    'non_payer_means': {c: float(v) for c, v in eng_non_payer.items()},
}

# ── 14. LTV30 buckets ──
ltv_all = df['ltv30'].dropna()
buckets = [0, 0.01, 1, 5, 10, 50, 100, 500, 1000, 5000, float('inf')]
labels = ['0', '0-1', '1-5', '5-10', '10-50', '50-100', '100-500', '500-1K', '1K-5K', '5K+']
df['ltv_bucket'] = pd.cut(ltv_all, bins=buckets, labels=labels, right=False)
bucket_dist = df.groupby('ltv_bucket', observed=True).size()
results['ltv30_bucket_distribution'] = {str(k): int(v) for k, v in bucket_dist.items()}

# ── 15. first_charge_day_offset distribution ──
fco = df[df['rev_d7'] > 0]['first_charge_day_offset_d7'].dropna()
results['first_charge_day_offset_d7_dist'] = {
    'mean': float(fco.mean()),
    'median': float(fco.median()),
    'value_counts': {str(k): int(v) for k, v in fco.value_counts().sort_index().head(10).items()},
}

print(json.dumps(results, indent=2, default=str))
