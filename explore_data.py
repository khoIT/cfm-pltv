import pandas as pd
import numpy as np

df = pd.read_csv('data/cfm_pltv_Dec_16_Feb_18.csv', nrows=50000)
print('Total rows sampled:', len(df))

payers = df[df['is_payer_30'] == 1]
print('Payer rate: %.1f%%' % (len(payers)/len(df)*100))
print('Payer LTV30 mean: {:,.0f}'.format(payers['ltv30'].mean()))
print('Payer LTV30 median: {:,.0f}'.format(payers['ltv30'].median()))
print()

# Revenue concentration
total_rev = df['ltv30'].sum()
for pct in [1, 5, 10, 20]:
    top_n = max(1, int(len(df) * pct / 100))
    top_rev = df['ltv30'].nlargest(top_n).sum()
    print('Top %d%% users = %.1f%% of revenue' % (pct, top_rev/total_rev*100))
print()

# Whale thresholds
for pct in [1, 5, 10]:
    thresh = df['ltv30'].quantile(1 - pct/100)
    whales = df[df['ltv30'] >= thresh]
    share = whales['ltv30'].sum() / total_rev * 100
    print('Top %d%% threshold: %s VND -> %.1f%% of LTV30' % (pct, '{:,.0f}'.format(thresh), share))
print()

# First charge timing
print('First charge day offset (payers):')
print(payers['first_charge_day_offset_d7'].describe())
d0 = (payers['first_charge_day_offset_d7'] == 0).sum()
d1_3 = ((payers['first_charge_day_offset_d7'] >= 1) & (payers['first_charge_day_offset_d7'] <= 3)).sum()
d4_7 = ((payers['first_charge_day_offset_d7'] >= 4) & (payers['first_charge_day_offset_d7'] <= 7)).sum()
print('D0 (same-day):', d0, '  D1-3:', d1_3, '  D4-7:', d4_7)
print()

# Media sources
print('Top media sources:')
print(df['media_source'].value_counts().head(8))
print()

# Countries
print('Top countries:')
print(df['first_country_code'].value_counts().head(8))
print()

# Gameplay stats
print('KD ratio describe:')
print(df['kd_d7'].describe())
print()
print('Win rate describe:')
print(df['win_rate_d7'].describe())
print()

# Multi-window D135 sample
df2 = pd.read_csv('data/cfm_pltv_D135_part01.csv', nrows=5000)
print('D135 columns:', list(df2.columns))
print()
print('window_days value counts:')
print(df2['window_days'].value_counts().sort_index())
print()

# Whale vs non-whale gameplay comparison
df_d7 = df2[df2['window_days'] == 7].copy()
df_d7['whale'] = df_d7['ltv30'] >= df_d7['ltv30'].quantile(0.95)
print('Whale vs non-whale D7 gameplay (top 5% by LTV30):')
cols = ['games', 'win_rate', 'kd', 'avg_score', 'max_level_seen', 'active_days', 'rev']
avail = [c for c in cols if c in df_d7.columns]
print(df_d7.groupby('whale')[avail].mean().T.to_string())
