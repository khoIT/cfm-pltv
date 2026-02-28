import pandas as pd

df_full = pd.read_csv('cfm_pltv_Feb22.csv')
df_sample = pd.read_csv('cfm_pltv_recent.csv')

print(f'Full: {len(df_full):,} rows')
print(f'Sample: {len(df_sample):,} rows ({len(df_sample)/len(df_full)*100:.1f}%)')
print(f'\nFull LTV30 stats:\n{df_full["ltv_30d"].describe()}')
print(f'\nSample LTV30 stats:\n{df_sample["ltv_30d"].describe()}')

# Check if it's just recent data or stratified
if 'install_date' in df_full.columns:
    print(f'\nFull date range: {df_full["install_date"].min()} to {df_full["install_date"].max()}')
    print(f'Sample date range: {df_sample["install_date"].min()} to {df_sample["install_date"].max()}')
