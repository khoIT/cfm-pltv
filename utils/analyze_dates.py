"""
Quick script to analyze date ranges in cfm_pltv.csv
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_path = DATA_DIR / "cfm_pltv.csv"

print("Loading data...")
df = pd.read_csv(csv_path, nrows=100_000)
df['install_date'] = pd.to_datetime(df['install_date'])

print(f"\n{'='*60}")
print(f"Date Range Analysis (first 100k rows)")
print(f"{'='*60}")
print(f"Install date range: {df['install_date'].min()} to {df['install_date'].max()}")
print(f"Total rows sampled: {len(df):,}")
print(f"Labeled (ltv30 not null): {df['ltv30'].notna().sum():,}")
print(f"Mean LTV30: ${df['ltv30'].mean():.2f}")
print(f"\nInstall date distribution:")
print(df['install_date'].value_counts().sort_index().tail(20))
