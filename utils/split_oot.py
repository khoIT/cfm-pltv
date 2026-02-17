"""
Split cfm_pltv_test_oot.csv into two test datasets by date.

Test 1: 2026-01-09 to 2026-01-13 (~118k rows) — closer to training window
Test 2: 2026-01-14 to 2026-01-18 (~82k rows)  — further out, harder test
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SPLIT_DATE = "2026-01-14"

def main():
    print("="*70)
    print("Splitting OOT Test Set into Test 1 and Test 2")
    print("="*70)
    print("Note: Run create_test_datasets.py first to generate cfm_pltv_test_oot.csv")

    df = pd.read_csv(DATA_DIR / "cfm_pltv_test_oot.csv", low_memory=False)
    df['install_date'] = pd.to_datetime(df['install_date'])
    print(f"\nOOT total: {len(df):,} rows  ({df['install_date'].min().date()} to {df['install_date'].max().date()})")

    t1 = df[df['install_date'] < SPLIT_DATE].copy()
    t2 = df[df['install_date'] >= SPLIT_DATE].copy()

    t1.to_csv(DATA_DIR / "cfm_pltv_test1.csv", index=False)
    t2.to_csv(DATA_DIR / "cfm_pltv_test2.csv", index=False)

    print(f"\nTest 1: {len(t1):,} rows  ({t1['install_date'].min().date()} to {t1['install_date'].max().date()})  → cfm_pltv_test1.csv")
    print(f"Test 2: {len(t2):,} rows  ({t2['install_date'].min().date()} to {t2['install_date'].max().date()})  → cfm_pltv_test2.csv")
    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()
