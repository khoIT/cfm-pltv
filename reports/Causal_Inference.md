# Causal Inference — CFM pLTV

## Business Context
For users with **rev_d7 = 0**, what behavioral signals predict late conversion (paying after D7)?
This analysis identifies **causal drivers** of late payment using observational data,
informing both feature engineering and product interventions.

**Key Question:** What causes a D7 non-payer to become a D30 payer? Can we intervene to increase conversion?

## Data Selection SQL (Trino/Iceberg)
```sql
-- Compare behavioral features between late payers and non-payers within D7=0 segment
WITH d7_zero AS (
  SELECT *, CASE WHEN ltv30 > 0 THEN 1 ELSE 0 END AS is_late_payer
  FROM cfm_pltv_features
  WHERE rev_d7 = 0
)
SELECT
  is_late_payer,
  AVG(games_d7) AS avg_games,
  AVG(active_days_d7) AS avg_active_days,
  AVG(kd_d7) AS avg_kd,
  AVG(login_rows_d7) AS avg_logins,
  COUNT(*) AS users
FROM d7_zero
GROUP BY is_late_payer
```

## Analytical Steps
1. Segment D7=0 users into late payers (ltv30 > 0) vs non-payers (ltv30 = 0)
2. Compare behavioral feature distributions (mean, ratio)
3. Bucket engagement features and compute conversion rates per bucket
4. Pseudo-causal analysis: compare high-engagement vs low-engagement groups
5. Rank features by discriminative power (payer/non-payer ratio)

## Key Charts

### 1. Feature Means: Late Payers vs Non-Payers
![Feature Comparison](plots/causal_feature_comparison.png)

### 2. Top Discriminating Features
![Discriminators](plots/causal_top_discriminators.png)

### 3. Late Conversion by Games Played
![Games Conversion](plots/causal_games_conversion.png)

### 4. Late Conversion by Active Days
![Active Days](plots/causal_active_days_conversion.png)

## Findings

### Feature Comparison (D7=0 segment: 1,596,341 users)
| Feature | Late Payer Mean | Non-Payer Mean | Ratio |
|---------|----------------|----------------|-------|
| kills_d7 | 769.982 | 205.457 | 3.75x |
| games_d7 | 66.006 | 19.594 | 3.37x |
| deaths_d7 | 399.867 | 119.901 | 3.33x |
| login_rows_d7 | 24.714 | 8.116 | 3.04x |
| max_level_seen_d7 | 44.096 | 18.081 | 2.44x |
| max_level_game_d7 | 44.967 | 19.774 | 2.27x |
| active_days_d7 | 5.580 | 2.753 | 2.03x |
| win_rate_d7 | 0.410 | 0.298 | 1.38x |
| kd_d7 | 2.147 | 1.657 | 1.30x |
| avg_game_duration_d7 | 373.568 | 298.175 | 1.25x |
| avg_score_d7 | 6816.970 | 5477.890 | 1.24x |
| max_ladderscore_d7 | 1148.567 | 963.524 | 1.19x |

### 1. Strongest Behavioral Predictor
- **kills_d7** has the highest payer/non-payer ratio at 3.75x
- Late payers show significantly different behavior even before paying

### 2. Engagement → Conversion Lift
- High engagement (games > median=5): **5.186%** late conversion
- Low engagement (games ≤ median): **0.585%** late conversion
- **Lift: 8.86x** — engaged non-payers are much more likely to convert

### 3. Dose-Response Pattern
- Late conversion rate increases monotonically with games played and active days
- This dose-response pattern strengthens the causal argument
- Users who play 50+ games in D7 have dramatically higher conversion rates

### 4. Implications for Causality
- **Limitation:** This is observational — we cannot prove engagement *causes* payment
- **Support for causality:** Dose-response relationship, temporal ordering (engagement precedes payment)
- **Confounders:** Intrinsic user quality, device quality, network effects

## Business Impact & Next Actions

1. **Engagement Nudges:** Push notifications to D7=0 users with moderate engagement to play more games
2. **Feature Engineering:** Prioritize kills_d7 and engagement features in ML models
3. **A/B Test Design:** Test engagement-boosting interventions (daily rewards, challenges)
   - Treatment: Engagement incentives to D7 non-payers
   - Control: No intervention
   - Metric: D30 payer rate, LTV30
4. **Targeted Offers:** D8-D14 discount offers to highly engaged non-payers
5. **Retention Priority:** Keep engaged non-payers active — they're the highest-potential late converters
