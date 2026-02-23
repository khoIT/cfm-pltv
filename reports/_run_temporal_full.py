"""One-off: temporal analysis on full cfm_pltv_2025_12_16.csv"""
import pandas as pd
import numpy as np

DATA = "/Users/lap16299/Documents/code/cfm-pltv/data/cfm_pltv_2025_12_16.csv"
print("Loading full dataset …")
df = pd.read_csv(DATA, low_memory=False)
df["install_date"] = pd.to_datetime(df["install_date"])
df["is_payer_d7"] = (df["rev_d7"] > 0).astype(int)
df["is_late_payer"] = ((df["rev_d7"] == 0) & (df["ltv30"] > 0)).astype(int)
df["rev_d8_d30"] = df["ltv30"] - df["rev_d7"]
print(f"Rows: {len(df):,}  |  Cols: {df.shape[1]}")
print(f"Install dates: {df.install_date.min().date()} → {df.install_date.max().date()}")
print(f"Unique dates: {df.install_date.nunique()}")
print()

# ── daily aggregation ─────────────────────────────────────────────
daily = df.groupby("install_date").agg(
    users       = ("vopenid", "count"),
    payers_d7   = ("is_payer_d7", "sum"),
    payers_d30  = ("is_payer_30", "sum"),
    late_payers = ("is_late_payer", "sum"),
    payer_rate_d30 = ("is_payer_30", "mean"),
    payer_rate_d7  = ("is_payer_d7", "mean"),
    late_payer_rate = ("is_late_payer", "mean"),
    mean_ltv30  = ("ltv30", "mean"),
    median_ltv30 = ("ltv30", "median"),
    mean_rev_d7 = ("rev_d7", "mean"),
    total_ltv30 = ("ltv30", "sum"),
    total_rev_d7 = ("rev_d7", "sum"),
    mean_games  = ("games_d7", "mean"),
    mean_active_days = ("active_days_d7", "mean"),
    mean_kd     = ("kd_d7", "mean"),
).reset_index()
daily["arpu_d30"] = daily["total_ltv30"] / daily["users"]
daily["arpu_d7"]  = daily["total_rev_d7"] / daily["users"]
daily["d7_to_d30_ratio"] = np.where(daily["total_ltv30"] > 0,
                                     daily["total_rev_d7"] / daily["total_ltv30"], 0)
daily["late_rev"] = daily["total_ltv30"] - daily["total_rev_d7"]
daily["install_dow"] = daily["install_date"].dt.day_name()

# ── Print full daily table ────────────────────────────────────────
print("=" * 120)
print("DAILY COHORT TABLE (full dataset)")
print("=" * 120)
for _, r in daily.iterrows():
    print(f"{r.install_date.date()} ({r.install_dow[:3]})  "
          f"Users: {int(r.users):>9,}  |  "
          f"D7 Payer%: {r.payer_rate_d7:5.2%}  D30 Payer%: {r.payer_rate_d30:5.2%}  "
          f"Late%: {r.late_payer_rate:5.2%}  |  "
          f"ARPU D7: {r.arpu_d7:>12,.0f}  ARPU D30: {r.arpu_d30:>12,.0f}  |  "
          f"D7/D30: {r.d7_to_d30_ratio:5.1%}  |  "
          f"Games: {r.mean_games:5.1f}  ActiveD: {r.mean_active_days:4.1f}  K/D: {r.mean_kd:4.2f}")

# ── Summary stats ─────────────────────────────────────────────────
print()
print("=" * 120)
print("AGGREGATE SUMMARY")
print("=" * 120)

total_users = len(df)
total_rev_d30 = df["ltv30"].sum()
total_rev_d7  = df["rev_d7"].sum()
total_late_rev = total_rev_d30 - total_rev_d7
total_payers_d30 = df["is_payer_30"].sum()
total_payers_d7  = df["is_payer_d7"].sum()
total_late_payers = df["is_late_payer"].sum()

print(f"Total users:         {total_users:>12,}")
print(f"Total revenue D30:   ₫{total_rev_d30:>18,.0f}")
print(f"Total revenue D7:    ₫{total_rev_d7:>18,.0f}")
print(f"Late revenue (D8-30):₫{total_late_rev:>18,.0f}  ({total_late_rev/total_rev_d30*100:.1f}% of D30)")
print(f"D7 payers:           {total_payers_d7:>12,}  ({total_payers_d7/total_users*100:.2f}%)")
print(f"D30 payers:          {total_payers_d30:>12,}  ({total_payers_d30/total_users*100:.2f}%)")
print(f"Late payers:         {total_late_payers:>12,}  ({total_late_payers/total_users*100:.2f}%)")
print(f"Overall ARPU D30:    ₫{total_rev_d30/total_users:>12,.0f}")
print(f"Overall ARPU D7:     ₫{total_rev_d7/total_users:>12,.0f}")
print()

# ── Whale analysis ────────────────────────────────────────────────
whale_thresh = df["ltv30"].quantile(0.99)
whales = df[df["ltv30"] >= whale_thresh]
print(f"Whale threshold (P99): ₫{whale_thresh:,.0f}")
print(f"Whales: {len(whales):,} users ({len(whales)/total_users*100:.2f}%)")
print(f"Whale revenue D30:   ₫{whales['ltv30'].sum():,.0f}  ({whales['ltv30'].sum()/total_rev_d30*100:.1f}% of total)")
print(f"Whale ARPU D30:      ₫{whales['ltv30'].mean():,.0f}")
print()

# ── Late payer economics ──────────────────────────────────────────
late = df[df["is_late_payer"] == 1]
print(f"Late payers: {len(late):,}")
print(f"Late payer avg LTV30:  ₫{late['ltv30'].mean():,.0f}")
print(f"Late payer total rev:  ₫{late['ltv30'].sum():,.0f}  ({late['ltv30'].sum()/total_rev_d30*100:.1f}% of total)")
print(f"Late payers who are whales: {len(late[late['ltv30'] >= whale_thresh]):,}  ({len(late[late['ltv30'] >= whale_thresh])/len(whales)*100:.1f}% of all whales)")
print()

# ── Day-of-week effect ────────────────────────────────────────────
# exclude first day (launch anomaly) and last day (partial)
mid = daily.iloc[1:-1]  
dow = mid.groupby("install_dow").agg(
    avg_users=("users", "mean"),
    avg_arpu_d30=("arpu_d30", "mean"),
    avg_payer_rate=("payer_rate_d30", "mean"),
).sort_values("avg_arpu_d30", ascending=False)
print("DAY-OF-WEEK (excl. launch & last day):")
for d, r in dow.iterrows():
    print(f"  {d:>10s}  Users: {r.avg_users:>9,.0f}  ARPU: ₫{r.avg_arpu_d30:>12,.0f}  Payer%: {r.avg_payer_rate:.2%}")
print()

# ── Media source by date ──────────────────────────────────────────
print("TOP 5 MEDIA SOURCES:")
ms = df.groupby("media_source").agg(
    users=("vopenid", "count"),
    arpu=("ltv30", "mean"),
    payer_rate=("is_payer_30", "mean"),
    late_rate=("is_late_payer", "mean"),
).sort_values("users", ascending=False).head(5)
for src, r in ms.iterrows():
    print(f"  {src:>25s}  Users: {int(r.users):>9,}  ARPU: ₫{r.arpu:>12,.0f}  "
          f"Payer%: {r.payer_rate:.2%}  Late%: {r.late_rate:.2%}")
print()

# ── Trend direction ───────────────────────────────────────────────
from scipy.stats import linregress
x = np.arange(len(daily))
for metric in ["arpu_d30", "payer_rate_d30", "late_payer_rate", "mean_games", "d7_to_d30_ratio"]:
    slope, intercept, r_value, p_value, _ = linregress(x, daily[metric])
    direction = "📈 UP" if slope > 0 else "📉 DOWN"
    print(f"Trend {metric:>20s}: {direction}  slope={slope:.4f}  R²={r_value**2:.3f}  p={p_value:.4f}")
