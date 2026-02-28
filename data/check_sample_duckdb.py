import duckdb

con = duckdb.connect(':memory:')

# Load both datasets
con.execute("CREATE TABLE full_data AS SELECT * FROM read_csv_auto('cfm_pltv_Feb22.csv', sample_size=-1)")
con.execute("CREATE TABLE sample_data AS SELECT * FROM read_csv_auto('cfm_pltv_recent.csv', sample_size=-1)")

# Compare sizes
full_count = con.execute("SELECT COUNT(*) FROM full_data").fetchone()[0]
sample_count = con.execute("SELECT COUNT(*) FROM sample_data").fetchone()[0]
print(f"Full: {full_count:,} rows")
print(f"Sample: {sample_count:,} rows ({sample_count/full_count*100:.1f}%)")

# Compare LTV30 distributions
print("\nFull LTV30 stats:")
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
print(f"  min={stats_full[0]:.2f}, q25={stats_full[1]:.2f}, median={stats_full[2]:.2f}, q75={stats_full[3]:.2f}, max={stats_full[4]:.2f}")
print(f"  mean={stats_full[5]:.2f}, std={stats_full[6]:.2f}")

print("\nSample LTV30 stats:")
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
print(f"  min={stats_sample[0]:.2f}, q25={stats_sample[1]:.2f}, median={stats_sample[2]:.2f}, q75={stats_sample[3]:.2f}, max={stats_sample[4]:.2f}")
print(f"  mean={stats_sample[5]:.2f}, std={stats_sample[6]:.2f}")

# Check date ranges
print("\nDate ranges:")
date_full = con.execute("SELECT MIN(install_date), MAX(install_date) FROM full_data").fetchone()
date_sample = con.execute("SELECT MIN(install_date), MAX(install_date) FROM sample_data").fetchone()
print(f"Full: {date_full[0]} to {date_full[1]}")
print(f"Sample: {date_sample[0]} to {date_sample[1]}")

con.close()
