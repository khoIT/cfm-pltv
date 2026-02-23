"""
Split cfm_pltv_2025_12_16.csv into 4 purpose-specific datasets.

Strategy:
  1. cfm_train.csv  — Mature cohorts Dec 16 – Jan 5 (training + local demo ~300MB)
  2. cfm_eval.csv   — Mature cohorts Jan 6 – Jan 22 (temporal holdout + prod demo ~100MB)
  3. cfm_demo.csv   — Stratified sample from all mature cohorts (prod demo ~100MB)
  4. cfm_predict.csv — Immature cohorts Jan 23 – Feb 21 (inference/scoring ~100MB)

Run:  python3 data/split_dataset.py
"""

import pandas as pd
import numpy as np
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DATA_DIR, 'cfm_pltv_2025_12_16.csv')
SEED = 42

def log(msg):
    print(f"[split] {msg}", file=sys.stderr)

def describe_split(name, df_split, df_full):
    n = len(df_split)
    payer_rate = df_split['is_payer_30'].mean() * 100
    avg_ltv = df_split['ltv30'].mean()
    date_min = df_split['install_date'].min().date()
    date_max = df_split['install_date'].max().date()
    pct_total = n / len(df_full) * 100
    log(f"  {name}: {n:>10,} rows ({pct_total:5.1f}%) | "
        f"dates: {date_min} → {date_max} | "
        f"payer rate: {payer_rate:.2f}% | avg ltv30: {avg_ltv:,.0f}")

def main():
    log(f"Loading {SRC} ...")
    df = pd.read_csv(SRC, quotechar='"', low_memory=False)
    log(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

    # Parse key columns
    df['install_date'] = pd.to_datetime(df['install_date'], errors='coerce')
    df['ltv30'] = pd.to_numeric(df['ltv30'], errors='coerce')
    df['rev_d7'] = pd.to_numeric(df['rev_d7'], errors='coerce')
    df['is_payer_30'] = pd.to_numeric(df['is_payer_30'], errors='coerce')

    # ── Define splits ──
    # Mature cutoff: Jan 22 (30 days before Feb 21)
    TRAIN_END = pd.Timestamp('2026-01-05')
    EVAL_START = pd.Timestamp('2026-01-06')
    MATURE_END = pd.Timestamp('2026-01-22')

    mask_train = df['install_date'] <= TRAIN_END
    mask_eval = (df['install_date'] >= EVAL_START) & (df['install_date'] <= MATURE_END)
    mask_mature = df['install_date'] <= MATURE_END
    mask_predict = df['install_date'] > MATURE_END

    df_train = df[mask_train].copy()
    df_eval = df[mask_eval].copy()
    df_predict = df[mask_predict].copy()

    # Demo: stratified sample from all mature data (~500K rows ≈ 100MB)
    # Stratify by is_payer_30 to preserve payer rate
    df_mature = df[mask_mature].copy()
    TARGET_DEMO_ROWS = 500_000

    # Stratified sampling: sample proportionally by payer status
    rng = np.random.RandomState(SEED)
    demo_indices = []
    for payer_val in [0, 1]:
        group = df_mature[df_mature['is_payer_30'] == payer_val]
        frac = TARGET_DEMO_ROWS / len(df_mature)
        n_sample = min(int(len(group) * frac), len(group))
        sampled = group.sample(n=n_sample, random_state=rng)
        demo_indices.append(sampled.index)
    df_demo = df_mature.loc[pd.Index(np.concatenate(demo_indices))].copy()
    # Shuffle
    df_demo = df_demo.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Report ──
    log("Split summary:")
    describe_split("cfm_train.csv   ", df_train, df)
    describe_split("cfm_eval.csv    ", df_eval, df)
    describe_split("cfm_demo.csv    ", df_demo, df)
    describe_split("cfm_predict.csv ", df_predict, df)

    total_unique = len(set(df_train.index) | set(df_eval.index) | set(df_predict.index))
    log(f"\n  Train + Eval + Predict cover {total_unique:,} unique rows "
        f"({'no' if total_unique == len(df) else 'HAS'} overlap with full dataset)")

    train_eval_overlap = set(df_train.index) & set(df_eval.index)
    log(f"  Train ∩ Eval overlap: {len(train_eval_overlap)} rows (should be 0)")

    # ── Write files ──
    for name, split_df in [
        ('cfm_train.csv', df_train),
        ('cfm_eval.csv', df_eval),
        ('cfm_demo.csv', df_demo),
        ('cfm_predict.csv', df_predict),
    ]:
        path = os.path.join(DATA_DIR, name)
        log(f"Writing {path} ...")
        split_df.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        log(f"  → {size_mb:.1f} MB")

    log("Done.")

if __name__ == '__main__':
    main()
