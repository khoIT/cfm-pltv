"""
Analyze the new cfm_pltv_from_2026_01_19.csv dataset for scoring
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_path = DATA_DIR / "cfm_pltv_from_2026_01_19.csv"

print("Loading scoring dataset...")
df = pd.read_csv(csv_path)
df['install_date'] = pd.to_datetime(df['install_date'])

print(f"\n{'='*60}")
print(f"Scoring Dataset Analysis")
print(f"{'='*60}")
print(f"Total rows: {len(df):,}")
print(f"Install date range: {df['install_date'].min()} to {df['install_date'].max()}")
print(f"\nColumn check:")
print(f"  - Has 'ltv30': {'ltv30' in df.columns}")
print(f"  - Has 'is_payer_30': {'is_payer_30' in df.columns}")

if 'ltv30' in df.columns:
    print(f"\nLTV30 stats:")
    print(f"  - Non-null: {df['ltv30'].notna().sum():,} ({df['ltv30'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  - Null: {df['ltv30'].isna().sum():,} ({df['ltv30'].isna().sum()/len(df)*100:.1f}%)")
    if df['ltv30'].notna().sum() > 0:
        print(f"  - Mean (non-null): ${df['ltv30'].mean():.2f}")

print(f"\nInstall date distribution (last 10 days):")
print(df['install_date'].value_counts().sort_index().tail(10))
