# Cohort Comparison — CFM pLTV

## Business Context
Compare user cohorts by **media source** and **OS** to identify which acquisition channels
deliver the highest LTV and which have the most late payers (ML opportunity).

**Key Question:** Which channels produce the best users? Where does late payer detection add the most value?

## Data Selection SQL (Trino/Iceberg)
```sql
SELECT
  media_source,
  first_os,
  COUNT(*) AS users,
  AVG(CASE WHEN is_payer_30 = 1 THEN 1.0 ELSE 0.0 END) AS payer_rate_d30,
  AVG(CASE WHEN rev_d7 = 0 AND ltv30 > 0 THEN 1.0 ELSE 0.0 END) AS late_payer_rate,
  AVG(ltv30) AS arpu_d30,
  AVG(games_d7) AS avg_games,
  AVG(active_days_d7) AS avg_active_days
FROM cfm_pltv_features
GROUP BY media_source, first_os
ORDER BY arpu_d30 DESC
```

## Analytical Steps
1. Aggregate by media source: users, payer rates, ARPU, engagement metrics
2. Compare D7 vs late payer rates per channel
3. Analyze engagement profiles (games, active days, K/D) by source
4. Compare iOS vs Android monetization patterns
5. Visualize LTV30 distributions to detect whale concentration

## Key Charts

### 1. ARPU by Media Source
![ARPU](plots/cohort_arpu_by_source.png)

### 2. Payer Rates by Source
![Payer Rates](plots/cohort_payer_rates_by_source.png)

### 3. Engagement Heatmap
![Engagement](plots/cohort_engagement_heatmap.png)

### 4. OS Comparison
![OS](plots/cohort_os_comparison.png)

### 5. LTV30 Distribution by Source
![LTV Box](plots/cohort_ltv30_boxplot.png)

## Findings

### Media Source Summary
| Source | Users | ARPU (₫) | D30 Payer % | Late Payer % | Games D7 |
|--------|-------|-----------|-------------|--------------|----------|
| Apple Search Ads | 186,513 | 42,163 | 7.90% | 3.57% | 30.4 |
| organic | 606,184 | 34,234 | 7.35% | 3.23% | 25.3 |
| Facebook Ads | 117,605 | 27,149 | 7.98% | 3.44% | 24.5 |
| tiktokglobal_int | 286,748 | 15,477 | 5.69% | 2.37% | 20.5 |
| googleadwords_int | 446,877 | 9,422 | 3.98% | 1.61% | 17.6 |

### 1. Best ARPU Channel
- **Apple Search Ads** leads with ARPU ₫42,163 and 7.90% D30 payer rate

### 2. Worst ARPU Channel
- **googleadwords_int** has lowest ARPU at ₫9,422

### 3. Highest Late Payer Opportunity
- **Apple Search Ads** has the highest late payer rate at 3.57%
- This channel benefits most from ML late payer detection

### 4. OS Differences
- iOS and Android show different monetization patterns
- iOS typically has higher ARPU but lower volume

### 5. Engagement ≠ Revenue
- High gameplay engagement (games_d7) doesn't always correlate with high ARPU
- Some channels bring engaged players who don't monetize

## Business Impact & Next Actions

1. **Budget Reallocation:** Shift UA budget toward channels with highest ARPU-adjusted ROI
2. **Channel-Specific Seeds:** Build separate lookalike seeds per media source for better targeting
3. **Late Payer Campaigns:** Target Apple Search Ads users with D8-D14 monetization nudges
4. **OS-Specific Offers:** Customize pricing and offers per platform
5. **Quality Monitoring:** Track ARPU by source weekly to detect degradation
