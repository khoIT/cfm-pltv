"""
Create two fixed test datasets from cfm_pltv.csv to prevent train/test leakage:

1. Test Dataset #1 (OOT Labeled Holdout): Last 10 days of labeled period
   - install_date: 2026-01-09 to 2026-01-18
   - Has full LTV30 labels
   - Out-of-time relative to training data
   - Used for model evaluation

2. Test Dataset #2 (Recent Unlabeled Scoring): Post-labeled period
   - install_date: 2026-01-19 onwards (if exists in future data)
   - No LTV30 labels (too recent)
   - Used for seed selection simulation

3. Training Dataset: Everything EXCEPT test set #1
   - install_date: 2025-12-16 to 2026-01-08
   - This becomes the max available training data
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# File paths (using correct iamount data)
FULL_DATA = DATA_DIR / "clm_pltv_iamount.csv"
TRAIN_DATA = DATA_DIR / "cfm_pltv_train.csv"
TEST_OOT = DATA_DIR / "cfm_pltv_test_oot.csv"
TEST_SCORING = DATA_DIR / "cfm_pltv_test_scoring.csv"

# Date boundaries
LABELED_END = "2026-01-18"  # Last day with LTV30 labels
OOT_START = "2026-01-09"     # Last 10 days for OOT test
SCORING_START = "2026-01-19" # Future installs for scoring (if any)

def main():
    print("="*70)
    print("Creating Fixed Test Datasets")
    print("="*70)
    
    # Load full data
    print(f"\n1. Loading full dataset: {FULL_DATA}")
    df = pd.read_csv(FULL_DATA)
    df['install_date'] = pd.to_datetime(df['install_date'])
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['install_date'].min()} to {df['install_date'].max()}")
    
    # Split 1: OOT Labeled Holdout (last 10 days)
    print(f"\n2. Creating Test Dataset #1 (OOT Labeled Holdout)")
    print(f"   Date range: {OOT_START} to {LABELED_END}")
    test_oot = df[(df['install_date'] >= OOT_START) & 
                  (df['install_date'] <= LABELED_END)].copy()
    print(f"   Rows: {len(test_oot):,}")
    print(f"   Labeled (ltv30 not null): {test_oot['ltv30'].notna().sum():,}")
    print(f"   Mean LTV30: ${test_oot['ltv30'].mean():.2f}")
    test_oot.to_csv(TEST_OOT, index=False)
    print(f"   ✅ Saved to: {TEST_OOT}")
    
    # Split 2: Training Data (everything before OOT test)
    print(f"\n3. Creating Training Dataset")
    print(f"   Date range: up to {pd.to_datetime(OOT_START) - pd.Timedelta(days=1)}")
    train = df[df['install_date'] < OOT_START].copy()
    print(f"   Rows: {len(train):,}")
    print(f"   Labeled (ltv30 not null): {train['ltv30'].notna().sum():,}")
    print(f"   Mean LTV30: ${train['ltv30'].mean():.2f}")
    train.to_csv(TRAIN_DATA, index=False)
    print(f"   ✅ Saved to: {TRAIN_DATA}")
    
    # Split 3: Recent Scoring Data (future installs, if any)
    print(f"\n4. Creating Test Dataset #2 (Recent Scoring - Unlabeled)")
    print(f"   Date range: {SCORING_START} onwards")
    test_scoring = df[df['install_date'] >= SCORING_START].copy()
    
    if len(test_scoring) > 0:
        print(f"   Rows: {len(test_scoring):,}")
        # Remove labels (simulate unlabeled future data)
        test_scoring['ltv30'] = None
        test_scoring['is_payer_30'] = None
        print(f"   Labels removed (simulating unlabeled future cohort)")
        test_scoring.to_csv(TEST_SCORING, index=False)
        print(f"   ✅ Saved to: {TEST_SCORING}")
    else:
        print(f"   ⚠️ No data after {SCORING_START} — skipping scoring dataset")
        print(f"   (This is expected if your data only goes to {LABELED_END})")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Training data:     {len(train):,} rows  → cfm_pltv_train.csv")
    print(f"Test OOT holdout:  {len(test_oot):,} rows  → cfm_pltv_test_oot.csv")
    if len(test_scoring) > 0:
        print(f"Test scoring:      {len(test_scoring):,} rows  → cfm_pltv_test_scoring.csv")
    print(f"\n✅ All datasets created successfully!")
    print(f"\nNext steps:")
    print(f"1. Update shared.py to load cfm_pltv_train.csv by default")
    print(f"2. Update pages to use cfm_pltv_test_oot.csv for evaluation")
    print(f"3. Users can now train on any subset of training data without leakage")

if __name__ == "__main__":
    main()
