# Temporal Analysis — CFM pLTV

## Business Context
Understanding how user quality evolves over the first week of CFM's launch in Vietnam.
Launch date: 2025-12-16. Data covers 7 install cohorts (Dec 16–22).

**Key Question:** Does user quality degrade as initial hype fades? When is the optimal window for UA investment?

## Data Selection SQL (Trino/Iceberg)
```sql
SELECT
  install_date,
  COUNT(*) AS users,
  AVG(CASE WHEN ltv30 > 0 THEN 1.0 ELSE 0.0 END) AS payer_rate_d30,
  AVG(CASE WHEN rev_d7 > 0 THEN 1.0 ELSE 0.0 END) AS payer_rate_d7,
  AVG(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1.0 ELSE 0.0 END) AS late_payer_rate,
  AVG(ltv30) AS arpu_d30,
  SUM(ltv30) AS total_revenue
FROM cfm_pltv_features
GROUP BY install_date
ORDER BY install_date
```

## Analytical Steps
1. Aggregate daily cohort metrics: user volume, payer rates (D7, D30, late), ARPU, engagement
2. Compute D7/D30 revenue ratio per cohort to measure early-vs-late monetization
3. Track engagement metrics (games, active days) for quality degradation signals
4. Identify inflection points and trends

## Key Charts

### 1. Daily Install Volume & Payer Rates
![Volume & Payer Rates](plots/temporal_daily_volume_payer_rates.png)

### 2. ARPU Trends (D7 vs D30)
![ARPU Trends](plots/temporal_arpu_trend.png)

### 3. D7 Revenue as % of D30
![D7/D30 Ratio](plots/temporal_d7_d30_ratio.png)

### 4. Engagement Trends
![Engagement](plots/temporal_engagement_trends.png)

## Findings

### 1. Massive Launch-Day Spike, Rapid Normalization
- **Launch day (Dec 16):** 435,015 installs — 3-4× higher than subsequent days
- **By Dec 22:** 23,793 installs (partial day)
- Organic installs dominate launch day; paid UA ramps up later

### 2. Payer Rate Trends
- **D30 payer rate:** Launch day at 10.51%, peak on 2025-12-16 at 10.51%
- **Late payer rate:** Ranges from 1.92% to 3.83%
- Later cohorts may show higher payer rates as organic "curious" installs fade and paid UA targets higher-intent users

### 3. D7/D30 Revenue Ratio
- D7 revenue captures **37.9%–41.8%** of D30 revenue
- Significant revenue accrues after D7, confirming the value of late payer detection

### 4. ARPU by Cohort
- ARPU D30 ranges from ₫530,359 to ₫1,359,007
- Launch-day ARPU may differ from steady-state due to organic user mix

## Business Impact & Next Actions

1. **UA Timing:** Later cohorts (Dec 18+) may show higher quality — invest in sustained UA, not just launch burst
2. **Late Payer Signal:** 2-3% late payer rate across all cohorts validates the ML late-payer detection approach
3. **Revenue Forecasting:** D7 captures only ~39% of D30 revenue — D30 forecasts must account for late revenue
4. **Cohort Monitoring:** Establish weekly cohort dashboards to detect quality degradation early
5. **Seasonal Effects:** Need more data (2+ months) to distinguish day-of-week from trend effects
