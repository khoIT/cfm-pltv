"""Combine the sharded CSV exports into a single cfm_pltv.csv."""
import pandas as pd
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

files = sorted(glob.glob(str(DATA_DIR / "_with_params*.csv")))
print(f"Found {len(files)} shard files")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Combined: {len(df):,} rows, {df.shape[1]} cols")

out_path = DATA_DIR / "cfm_pltv.csv"
df.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved {out_path}  ({size_mb:.1f} MB)")
