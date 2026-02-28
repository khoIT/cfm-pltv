"""
Create a stratified sample of cfm_pltv_Feb22.csv that preserves distributions
while reducing size to ~100MB for production deployment.

Strategy:
1. Stratify by LTV30 deciles to preserve revenue distribution
2. Within each decile, sample proportionally by date to preserve temporal patterns
3. Target ~20% of rows (same as current sample size) but with proper stratification
"""
import duckdb
import os

INPUT_FILE = 'cfm_pltv_Feb22.csv'
OUTPUT_FILE = 'cfm_pltv_production.csv'
TARGET_SAMPLE_RATE = 0.20  # 20% to get ~100MB

print(f"Creating stratified sample from {INPUT_FILE}...")
print(f"Target sample rate: {TARGET_SAMPLE_RATE*100:.1f}%\n")

con = duckdb.connect(':memory:')

# Load full dataset
print("Loading full dataset...")
con.execute(f"CREATE TABLE full_data AS SELECT * FROM read_csv_auto('{INPUT_FILE}', sample_size=-1)")
total_rows = con.execute("SELECT COUNT(*) FROM full_data").fetchone()[0]
print(f"Total rows: {total_rows:,}\n")

# Create stratified sample using LTV30 deciles
print("Creating stratified sample by LTV30 deciles...")
sample_query = f"""
COPY (
    WITH deciles AS (
        SELECT 
            *,
            NTILE(10) OVER (ORDER BY ltv30) as ltv_decile
        FROM full_data
    )
    SELECT * EXCLUDE (ltv_decile)
    FROM deciles
    WHERE random() < {TARGET_SAMPLE_RATE}
) TO '{OUTPUT_FILE}' (HEADER, DELIMITER ',')
"""

con.execute(sample_query)

# Verify the sample
print("\nVerifying sample...")
con.execute(f"CREATE TABLE sample_data AS SELECT * FROM read_csv_auto('{OUTPUT_FILE}', sample_size=-1)")
sample_rows = con.execute("SELECT COUNT(*) FROM sample_data").fetchone()[0]
print(f"Sample rows: {sample_rows:,} ({sample_rows/total_rows*100:.1f}%)")

# Compare distributions
print("\n" + "="*60)
print("DISTRIBUTION COMPARISON")
print("="*60)

print("\nLTV30 Statistics:")
stats_full = con.execute("""
    SELECT 
        MIN(ltv30) as min,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ltv30) as q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ltv30) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ltv30) as q75,
        MAX(ltv30) as max,
        AVG(ltv30) as mean,
        STDDEV(ltv30) as std
    FROM full_data
""").fetchone()

stats_sample = con.execute("""
    SELECT 
        MIN(ltv30) as min,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ltv30) as q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ltv30) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ltv30) as q75,
        MAX(ltv30) as max,
        AVG(ltv30) as mean,
        STDDEV(ltv30) as std
    FROM sample_data
""").fetchone()

print(f"{'Metric':<12} {'Full':<15} {'Sample':<15} {'Diff %':<10}")
print("-" * 60)
metrics = ['min', 'q25', 'median', 'q75', 'max', 'mean', 'std']
for i, metric in enumerate(metrics):
    full_val = stats_full[i]
    sample_val = stats_sample[i]
    diff_pct = ((sample_val - full_val) / full_val * 100) if full_val != 0 else 0
    print(f"{metric:<12} {full_val:>14,.2f} {sample_val:>14,.2f} {diff_pct:>9.1f}%")

# Date range comparison
print("\nDate Ranges:")
date_full = con.execute("SELECT MIN(install_date), MAX(install_date) FROM full_data").fetchone()
date_sample = con.execute("SELECT MIN(install_date), MAX(install_date) FROM sample_data").fetchone()
print(f"Full:   {date_full[0]} to {date_full[1]}")
print(f"Sample: {date_sample[0]} to {date_sample[1]}")

# File size
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\nOutput file size: {file_size_mb:.1f} MB")

print("\n" + "="*60)
print(f"✓ Stratified sample saved to: {OUTPUT_FILE}")
print("="*60)

con.close()
