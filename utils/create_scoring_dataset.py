"""
Create Test Dataset #2 (Scoring Dataset) from cfm_pltv_from_2026_01_19.csv

This simulates a "score users now" scenario where:
- Users installed recently (2026-01-19 onwards)
- We DON'T know their LTV30 yet (labels removed)
- We want to rank them for seed selection using the trained model
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# File paths (using correct iamount data)
SOURCE_FILE = DATA_DIR / "clm_pltv_from_2026_01_19_iamount.csv"
OUTPUT_FILE = DATA_DIR / "cfm_pltv_test_scoring.csv"

def main():
    print("="*70)
    print("Creating Test Dataset #2 (Scoring Dataset)")
    print("="*70)
    
    # Load source data
    print(f"\n1. Loading source: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)
    df['install_date'] = pd.to_datetime(df['install_date'])
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['install_date'].min()} to {df['install_date'].max()}")
    
    # Check current labels
    if 'ltv30' in df.columns:
        print(f"   Current LTV30: {df['ltv30'].notna().sum():,} labeled rows (mean: ${df['ltv30'].mean():.2f})")
    
    # Remove labels to simulate unlabeled future cohort
    print(f"\n2. Removing labels (simulating 'score users now' scenario)")
    df_scoring = df.copy()
    
    # Keep the ground truth in a separate column for later evaluation (optional)
    if 'ltv30' in df.columns:
        df_scoring['ltv30_ground_truth'] = df['ltv30']  # Hidden ground truth for evaluation
        df_scoring['is_payer_30_ground_truth'] = df['is_payer_30']
    
    # Remove the actual labels that the model would see
    df_scoring['ltv30'] = None
    df_scoring['is_payer_30'] = None
    
    print(f"   ✅ Removed 'ltv30' and 'is_payer_30' columns")
    print(f"   ✅ Kept ground truth as 'ltv30_ground_truth' for evaluation")
    
    # Save
    print(f"\n3. Saving scoring dataset")
    df_scoring.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    print(f"   Rows: {len(df_scoring):,}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Scoring dataset: {len(df_scoring):,} rows → cfm_pltv_test_scoring.csv")
    print(f"Date range: {df_scoring['install_date'].min()} to {df_scoring['install_date'].max()}")
    print(f"Labels removed: ltv30, is_payer_30 (set to None)")
    print(f"Ground truth preserved: ltv30_ground_truth, is_payer_30_ground_truth")
    print(f"\n✅ Scoring dataset created successfully!")
    print(f"\nUse case:")
    print(f"  - Train model on cfm_pltv_train.csv (2025-12-16 to 2026-01-08)")
    print(f"  - Evaluate on cfm_pltv_test_oot.csv (2026-01-09 to 2026-01-18)")
    print(f"  - Score NEW users on cfm_pltv_test_scoring.csv (2026-01-19 to 2026-02-16)")
    print(f"  - Export Top-K for seed selection (no labels needed)")

if __name__ == "__main__":
    main()
