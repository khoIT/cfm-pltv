# CFM pLTV Raw Data Analysis

**Source file:** `data/cfm_pltv_2025_12_16.csv`
**Analysis date:** 2025-02-23
**Game:** CFM Vietnam (`cfm_vn`)

---

## 0. Query
``` sql 
WITH params AS ( SELECT DATE '2025-12-16' AS data_start, 7 AS feat_days, 30 AS label_days ), 
/* 1) UA cohort */ 
ua_cohort AS ( SELECT vopenid, CAST(install_time AS date) AS install_date, game_id, media_source, campaign_id, adset_id, ad_id, site_id, first_os, last_os, 
					first_country_code, last_country_code, first_login_channel, last_login_channel 
				FROM iceberg.cfm_vn.std_master_user_profile 
				WHERE vopenid IS NOT NULL 
				AND CAST(install_time AS date) >= (SELECT data_start FROM params) 
				-- AND CAST(install_time AS date) <= date_add('day', -(SELECT label_days FROM params), current_date) 
				), 
/* 2) role mapping (optional, not used for gameplay) */ 
role_map AS ( SELECT vopenid, min_by(roleid, ds) AS roleid 
				FROM iceberg.cfm_vn.etl_new_register 
				WHERE ds >= (SELECT data_start FROM params) 
				AND vopenid IS NOT NULL 
				AND roleid IS NOT NULL GROUP BY 1 ), 
base AS ( SELECT u.*, rm.roleid 
			FROM ua_cohort u LEFT JOIN role_map rm ON u.vopenid = rm.vopenid ), 
/* 3) Login features D0–D7 */ 
login_d7 AS ( SELECT b.vopenid, b.install_date, COUNT(*) AS login_rows_d7, COUNT(DISTINCT CAST(l.dteventtime AS date)) AS active_days_d7, 
				approx_distinct(NULLIF(l.loginchannel, '')) AS loginchannel_variety_d7, approx_distinct(NULLIF(l.network, '')) AS network_variety_d7, 
				approx_distinct(NULLIF(l.clientversion, '')) AS clientversion_variety_d7, MAX(TRY_CAST(l.level AS integer)) AS max_level_seen_d7, 
				MAX(TRY_CAST(l.viplevel AS integer)) AS max_viplevel_seen_d7, MAX(TRY_CAST(l.ladderscore AS double)) AS max_ladderscore_d7 
				FROM base b JOIN iceberg.cfm_vn.etl_login l ON l.vopenid = b.vopenid 
				AND l.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) 
				AND CAST(l.dteventtime AS date) BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) 
				GROUP BY 1,2 ), 
/* 4) Gameplay features D0–D7 (JOIN via playeropenid) */ 
game_d7 AS ( SELECT b.vopenid, b.install_date, COUNT(*) AS games_d7, AVG(CASE WHEN TRY_CAST(g.gameresult AS integer) = 1 THEN 1.0 ELSE 0.0 END) AS win_rate_d7, 
				AVG(TRY_CAST(g.gameduration AS double)) AS avg_game_duration_d7, AVG(TRY_CAST(g.score AS double)) AS avg_score_d7, SUM(COALESCE(TRY_CAST(g.timeskill AS double), 0)) AS kills_d7, 
				SUM(COALESCE(TRY_CAST(g.timesbekilled AS double), 0)) AS deaths_d7, SUM(COALESCE(TRY_CAST(g.timesassists AS double), 0)) AS assists_d7, 
				(SUM(COALESCE(TRY_CAST(g.timeskill AS double), 0)) * 1.0) / NULLIF(SUM(COALESCE(TRY_CAST(g.timesbekilled AS double), 0)), 0) AS kd_d7, 
				MAX(TRY_CAST(g.level AS integer)) AS max_level_game_d7, MAX(TRY_CAST(g.ladderlevel AS double)) AS max_ladderlevel_d7 
				FROM base b JOIN iceberg.cfm_vn.etl_game_detail g ON g.playeropenid = b.vopenid 
				AND g.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) 
				GROUP BY 1,2 ), 
/* 5) Payment features + label using imoney_source */ 
pay_agg AS ( SELECT b.vopenid, b.install_date, SUM( CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) THEN COALESCE(TRY_CAST(p.imoney_source AS double), 0) / 100.0 ELSE 0 END ) AS rev_d7, 
			SUM( CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) THEN 1 ELSE 0 END ) AS txn_cnt_d7, 
			MIN( CASE WHEN COALESCE(TRY_CAST(p.imoney_source AS double), 0) / 100.0 > 0 AND p.ds BETWEEN b.install_date AND date_add('day', (SELECT feat_days FROM params) - 1, b.install_date) THEN date_diff('day', b.install_date, p.ds) ELSE NULL END ) AS first_charge_day_offset_d7,
			-- LTV30 label 
			SUM( CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params) - 1, b.install_date) THEN COALESCE(TRY_CAST(p.imoney_source AS double), 0) / 100.0 ELSE 0 END ) AS ltv30, 
			CASE WHEN SUM( CASE WHEN p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params) - 1, b.install_date) THEN COALESCE(TRY_CAST(p.imoney_source AS double), 0) / 100.0 ELSE 0 END ) > 0 THEN 1 ELSE 0 END AS is_payer_30 
			FROM base b LEFT JOIN iceberg.cfm_vn.etl_recharge p ON p.vopenid = b.vopenid 
			AND p.ds BETWEEN b.install_date AND date_add('day', (SELECT label_days FROM params) - 1, b.install_date) 
			GROUP BY 1,2 ) 
SELECT b.vopenid, b.roleid, b.install_date, b.game_id, b.media_source, b.campaign_id, b.adset_id, b.ad_id, b.site_id, b.first_os, b.last_os, b.first_country_code, 
		b.last_country_code, b.first_login_channel, b.last_login_channel, COALESCE(l.login_rows_d7, 0) AS login_rows_d7, COALESCE(l.active_days_d7, 0) AS active_days_d7, 
		COALESCE(l.loginchannel_variety_d7, 0) AS loginchannel_variety_d7, COALESCE(l.network_variety_d7, 0) AS network_variety_d7, COALESCE(l.clientversion_variety_d7, 0) AS clientversion_variety_d7, 
		COALESCE(l.max_level_seen_d7, 0) AS max_level_seen_d7, COALESCE(l.max_ladderscore_d7, 0) AS max_ladderscore_d7, 
		COALESCE(g.games_d7, 0) AS games_d7, COALESCE(g.win_rate_d7, 0) AS win_rate_d7, COALESCE(g.avg_game_duration_d7, 0) AS avg_game_duration_d7, 
		COALESCE(g.avg_score_d7, 0) AS avg_score_d7, COALESCE(g.kills_d7, 0) AS kills_d7, COALESCE(g.deaths_d7, 0) AS deaths_d7, COALESCE(g.assists_d7, 0) AS assists_d7, COALESCE(g.kd_d7, 0) AS kd_d7, 
		COALESCE(g.max_level_game_d7, 0) AS max_level_game_d7, COALESCE(g.max_ladderlevel_d7, 0) AS max_ladderlevel_d7, COALESCE(p.rev_d7, 0) AS rev_d7, COALESCE(p.txn_cnt_d7, 0) AS txn_cnt_d7, p.first_charge_day_offset_d7, 
		COALESCE(p.ltv30, 0) AS ltv30, COALESCE(p.is_payer_30, 0) AS is_payer_30 
FROM base b LEFT JOIN login_d7 l ON b.vopenid = l.vopenid 
AND b.install_date = l.install_date LEFT JOIN game_d7 g ON b.vopenid = g.vopenid 
AND b.install_date = g.install_date LEFT JOIN pay_agg p ON b.vopenid = p.vopenid 
AND b.install_date = p.install_date
```

## 1. Data Shape & Schema

| Metric | Value |
|--------|-------|
| **Rows** | 2,624,049 |
| **Columns** | 37 |
| **Country** | 100% Vietnam (VN) |
| **Game** | `cfm_vn` (single game) |

### Column Groups

| Group | Columns | Description |
|-------|---------|-------------|
| **Identity** | `vopenid`, `roleid` | Player and role identifiers |
| **Install** | `install_date`, `game_id` | Install cohort date, game ID |
| **Attribution** | `media_source`, `campaign_id`, `adset_id`, `ad_id`, `site_id` | UA attribution fields |
| **Device/Geo** | `first_os`, `last_os`, `first_country_code`, `last_country_code` | OS and geography |
| **Login behavior (d7)** | `first_login_channel`, `last_login_channel`, `login_rows_d7`, `active_days_d7`, `loginchannel_variety_d7`, `network_variety_d7`, `clientversion_variety_d7` | Login patterns in first 7 days |
| **Gameplay (d7)** | `max_level_seen_d7`, `max_ladderscore_d7`, `games_d7`, `win_rate_d7`, `avg_game_duration_d7`, `avg_score_d7`, `kills_d7`, `deaths_d7`, `assists_d7`, `kd_d7`, `max_level_game_d7`, `max_ladderlevel_d7` | In-game engagement metrics at 7 days |
| **Revenue (d7)** | `rev_d7`, `txn_cnt_d7`, `first_charge_day_offset_d7` | Revenue and transaction data at 7 days |
| **Target** | `ltv30` | 30-day lifetime value (VND) |
| **Label** | `is_payer_30` | Binary payer flag at 30 days |

### Notable Null Rates

| Column | Null % |
|--------|--------|
| `campaign_id` | 41.38% |
| `adset_id` | 60.50% |
| `ad_id` | 87.15% |
| `site_id` | 95.50% |
| `first_login_channel` | 57.22% |
| `last_login_channel` | 57.22% |
| `first_charge_day_offset_d7` | 97.34% (only populated for 7d payers) |

**Observation:** Attribution detail columns have high null rates — organic users and some ad networks don't populate `campaign_id`/`adset_id`/`ad_id`. The `site_id` is almost entirely empty. `first_login_channel` is missing for ~57% of users.

---

## 2. Date Range & Cohort Coverage

| Metric | Value |
|--------|-------|
| **Earliest install_date** | 2025-12-16 |
| **Latest install_date** | 2026-02-21 |
| **Span** | 68 unique days (~9.5 weeks) |

### Daily Install Volume

```
2025-12-16: 435,107  ← Launch spike
2025-12-17: 149,467
2025-12-18: 120,124
2025-12-19: 101,504
...
2025-12-25:  43,874  ← Gradual decay
...
2026-01-14:  18,277  ← Steady state
2026-01-19:  71,031  ← Campaign spike
2026-01-30:  47,480  ← Campaign spike
2026-02-09:  58,839  ← Campaign spike (Tet?)
2026-02-12:  48,897  ← Campaign spike
...
2026-02-21:   8,722  ← Most recent
```

**Key patterns:**
- **Massive launch day** (Dec 16): 435K installs, then rapid decay to ~100K/day within a few days
- **Steady-state** around mid-January: ~15-25K/day
- **Several campaign spikes** visible (Jan 19, Jan 30, Feb 9, Feb 12) likely corresponding to UA pushes or in-game events
- **LTV30 data validity**: Only cohorts installed before ~Jan 22 have full 30-day maturation. Cohorts after that have incomplete LTV30 data — this is a potential data quality concern.

---

## 3. Overall LTV & Payer Distribution

| Metric | Value |
|--------|-------|
| **Total users** | 2,624,049 |
| **Total payers (30d)** | 122,737 (4.68%) |
| **Non-payers** | 2,501,312 (95.32%) |
| **Mean LTV30** | 18,743 VND (~$0.75) |
| **Median LTV30** | 0 VND (non-payer) |
| **Max LTV30** | 74,730,000 VND (~$2,989) |
| **Total Revenue** | ~49.2B VND |

### LTV30 Bucket Distribution

| Bucket | Count | % of Total | Total Revenue (VND) | % of Revenue |
|--------|-------|-----------|---------------------|-------------|
| 0 (Non-payer) | 2,501,312 | 95.32% | 0 | 0% |
| 1 – 9K | 28,448 | 1.08% | 256M | 0.52% |
| 9K – 50K | 20,684 | 0.79% | 591M | 1.20% |
| 50K – 129K | 17,224 | 0.66% | 2.06B | 4.19% |
| 129K – 274K | 27,156 | 1.03% | 5.84B | 11.88% |
| 274K – 608K | 16,971 | 0.65% | 7.10B | 14.44% |
| 608K – 1.3M | 6,125 | 0.23% | 5.29B | 10.76% |
| 1.3M – 5.76M | 4,903 | 0.19% | 13.47B | 27.40% |
| 5.76M+ | 1,226 | 0.05% | 14.56B | 29.61% |

**Extremely heavy tail.** The top 0.05% of users (1,226 super whales) generate 29.6% of all revenue.

---

## 4. Whale Player Distribution

Whale tiers defined by LTV30 percentiles **among payers only**:

| Segment | LTV30 Range (VND) | Count | % of Payers | Total Revenue | % of Revenue | Avg LTV30 |
|---------|-------------------|-------|-------------|---------------|-------------|-----------|
| **Minnow** (≤p50) | 9K – 129K | 66,356 | 54.06% | 2.91B | 5.91% | 43,788 |
| **Dolphin** (p50–p90) | 133K – 608K | 44,127 | 35.95% | 12.95B | 26.32% | 293,389 |
| **Whale** (p90–p99) | 609K – 5.76M | 11,028 | 8.99% | 18.77B | 38.16% | 1,701,650 |
| **Super Whale** (>p99) | 5.77M – 74.73M | 1,226 | 1.00% | 14.56B | 29.61% | 11,879,335 |

### Payer LTV30 Percentiles

| Percentile | LTV30 (VND) |
|-----------|-------------|
| p10 | 9,000 |
| p25 | 18,000 |
| p50 | 129,000 |
| p75 | 274,000 |
| p90 | 608,000 |
| p95 | 1,304,000 |
| p99 | 5,761,000 |

**Key takeaway:** Classic 80/20 distribution on steroids:
- **Top 10% of payers** (Whales + Super Whales) = **67.8% of revenue**
- **Bottom 54% of payers** (Minnows) = only **5.9% of revenue**
- Predicting who becomes a whale is far more valuable than predicting payer/non-payer

---

## 5. Payer at 7D vs LTV at 30D — Correlation Analysis

### Cross-tabulation: 7d Payer → 30d Payer

| | Non-payer 30d | Payer 30d |
|---|---|---|
| **Non-payer 7d** | 2,501,312 | 53,032 |
| **Payer 7d** | 0 | 69,705 |

**Critical finding:**
- **100% of 7d payers become 30d payers** (by definition — rev_d7 > 0 means ltv30 ≥ rev_d7)
- **2.08% of 7d non-payers convert to 30d payers** (53,032 late converters)
- These late converters represent **43.2% of all 30d payers** — a very significant segment

### Revenue Comparison

| Segment | Count | Avg LTV30 | Median LTV30 |
|---------|-------|-----------|-------------|
| **7d payers** | 69,705 | 494,181 VND | 129,000 VND |
| **7d non-payers who pay by 30d** | 53,032 | 277,850 VND | 129,000 VND |
| **7d non-payers (all)** | 2,554,344 | 5,769 VND | 0 VND |

### Correlation Coefficients

| Metric | Pearson r |
|--------|-----------|
| `rev_d7` ↔ `ltv30` (all users) | **0.694** |
| `rev_d7` ↔ `ltv30` (among 7d payers only) | **0.728** |

### LTV30 / rev_d7 Ratio (among 7d payers)

| Stat | Ratio |
|------|-------|
| Mean | 4.20x |
| Median | 1.00x |
| p25 | 1.00x |
| p75 | 1.68x |

**Interpretation:**
- **r = 0.694** is a strong positive correlation. `rev_d7` is by far the most predictive single feature for `ltv30`.
- Median ratio of 1.0x means **half of 7d payers don't spend any more after day 7**. But the mean of 4.2x shows the other half can spend significantly more.
- **43% of 30d payers haven't paid by day 7** — a model relying only on `rev_d7` misses these entirely. Behavioral features (engagement, gameplay) are critical for identifying future converters.

---

## 6. Feature Correlations with LTV30

Ranked by absolute Pearson correlation:

| Feature | Correlation | Interpretation |
|---------|------------|---------------|
| `rev_d7` | **0.694** | Strongest single predictor |
| `txn_cnt_d7` | **0.462** | Transaction count matters |
| `login_rows_d7` | 0.139 | Login frequency |
| `games_d7` | 0.132 | Games played |
| `kills_d7` | 0.126 | Combat engagement |
| `max_ladderlevel_d7` | 0.122 | Competitive progression |
| `deaths_d7` | 0.114 | Time in combat (correlated with kills) |
| `max_level_seen_d7` | 0.114 | Level progression |
| `assists_d7` | 0.113 | Team play |
| `max_level_game_d7` | 0.112 | Game-level progression |
| `active_days_d7` | 0.088 | Retention signal |
| `network_variety_d7` | 0.065 | Multi-network usage |
| `avg_score_d7` | 0.039 | Performance |
| `avg_game_duration_d7` | 0.036 | Session length |
| `win_rate_d7` | 0.035 | Skill level |
| `max_ladderscore_d7` | 0.030 | Ladder ranking |
| `clientversion_variety_d7` | 0.022 | App updates |
| `loginchannel_variety_d7` | 0.022 | Login channel diversity |
| `kd_d7` | 0.012 | K/D ratio (weak signal) |
| `first_charge_day_offset_d7` | **-0.023** | Earlier charge → slightly higher LTV |

### Feature Groups by Signal Strength

- **Tier 1 (strong):** `rev_d7`, `txn_cnt_d7` — direct spend signals
- **Tier 2 (moderate):** `login_rows_d7`, `games_d7`, `kills_d7`, `max_ladderlevel_d7`, `max_level_seen_d7` — engagement/progression
- **Tier 3 (weak):** `active_days_d7`, `network_variety_d7`, game performance metrics — supplementary signals
- **Negative:** `first_charge_day_offset_d7` — weak negative means players who pay earlier tend to have slightly higher LTV

---

## 7. Engagement by Payer Status

| Feature | Payer Mean | Non-Payer Mean | Ratio |
|---------|-----------|---------------|-------|
| `active_days_d7` | 5.75 | 2.33 | 2.5x |
| `games_d7` | 72.15 | 15.61 | 4.6x |
| `max_level_seen_d7` | 48.37 | 14.42 | 3.4x |
| `avg_game_duration_d7` | 387.0 | 257.9 | 1.5x |
| `kills_d7` | 830.6 | 159.4 | 5.2x |

**Payers are dramatically more engaged:**
- Play **4.6x more games** in first 7 days
- Have **5.2x more kills** (deep FPS engagement)
- Are active **2.5x more days**
- Reach **3.4x higher levels**

These engagement features are especially valuable for predicting **late converters** (7d non-payers who pay by 30d).

---

## 8. Media Source Breakdown

| Source | Installs | Payer Rate | Avg LTV30 | Total Rev (VND) |
|--------|---------|-----------|-----------|----------------|
| **organic** | 1,020,192 | 5.28% | 24,842 | 25.3B |
| **googleadwords_int** | 840,060 | 2.64% | 6,257 | 5.3B |
| **tiktokglobal_int** | 335,563 | 5.24% | 14,191 | 4.8B |
| **Apple Search Ads** | 264,922 | 6.75% | 35,987 | 9.5B |
| **Facebook Ads** | 134,251 | 7.40% | 25,336 | 3.4B |
| **QR_code** | 3,059 | 7.68% | 69,397 | 212M |
| **tiktoklive_int** | 3,034 | 7.65% | 59,136 | 179M |
| **CFL_LPPre-reg** | 1,948 | 10.83% | 45,893 | 89M |
| **CFL_OB_Download** | 1,084 | 10.24% | 111,970 | 121M |

**Key observations:**
- **Google Ads** has the highest volume but **worst payer rate (2.64%)** and lowest avg LTV — likely broad targeting
- **Apple Search Ads** has best quality among major sources: 6.75% payer rate, highest avg LTV (35,987)
- **Facebook Ads** is second-best quality among major sources (7.40% payer rate)
- **Organic** is the largest source and reasonably high quality (5.28%)
- Small sources like **CFL_OB_Download** and **QR_code** show very high LTV but tiny volume

---

## 9. OS Breakdown

| OS | Count | Payer Rate | Avg LTV30 |
|----|-------|-----------|-----------|
| **Android** | 1,476,866 (56.3%) | 3.15% | 8,257 VND |
| **iOS** | 1,147,154 (43.7%) | 6.65% | 32,240 VND |

**iOS users monetize 3.9x higher** than Android users with **2.1x the payer rate**. This is a critical segmentation feature for LTV modeling.

---

## 10. First Charge Day Offset

Among 7d payers, when they first pay:

| Day | Count | % of 7d Payers |
|-----|-------|----------------|
| Day 0 (install day) | 24,494 | 35.1% |
| Day 1 | 13,286 | 19.1% |
| Day 2 | 8,673 | 12.4% |
| Day 3 | 6,868 | 9.9% |
| Day 4 | 5,962 | 8.6% |
| Day 5 | 5,515 | 7.9% |
| Day 6 | 4,907 | 7.0% |

**35% of payers convert on day 0**, with a smooth decay thereafter. Mean offset is 1.9 days.

---

## 11. Key Takeaways for LTV Modeling

### Data Quality Considerations
1. **Incomplete LTV30 for recent cohorts:** Installs after ~Jan 22 haven't matured 30 days. If the target is naively populated, these users will appear as non-payers but may still convert. **Filter to cohorts with full 30-day maturation for training.**
2. **High null rate in attribution fields:** `ad_id` (87% null), `site_id` (95.5% null) — these may need imputation or should be dropped. `media_source` and `campaign_id` are more usable.
3. **`first_login_channel`** is 57% null — investigate why and whether it's informative.

### Modeling Strategy Recommendations
4. **Two-stage model may outperform single model:**
   - **Stage 1:** Predict payer vs non-payer (classification)
   - **Stage 2:** Predict LTV30 amount among predicted payers (regression)
   - Reason: 95.3% are non-payers, and the value distribution among payers is extremely skewed.

5. **Feature engineering opportunities:**
   - `rev_d7` is the single best feature (r=0.694) but misses 43% of eventual payers
   - Engagement features (`games_d7`, `kills_d7`, `max_level_seen_d7`) are key for identifying late converters
   - `first_os` (iOS vs Android) is a strong segmentation variable (3.9x LTV difference)
   - `media_source` captures acquisition quality differences
   - Interaction features: `rev_d7 × active_days_d7`, `games_d7 × max_level_seen_d7`

6. **Whale prediction is high-value:**
   - Top 1% of payers = 29.6% of revenue
   - Top 10% of payers = 67.8% of revenue
   - Consider a separate whale classifier or ordinal model

7. **Late converter identification:**
   - 53,032 users (43% of payers) don't pay until after day 7
   - Their median LTV is 129K VND — same as early payers
   - Engagement signals are the primary way to identify them before they pay

### Currency Note
All monetary values are in **VND** (Vietnamese Dong). 1 USD ≈ 25,000 VND. So median payer LTV30 of 129,000 VND ≈ $5.16 USD.

---

## 12. Dataset Splits

Source file (495MB, 2.6M rows) split into 4 purpose-specific files using **temporal splitting** on mature cohorts + stratified sampling for demo.

**Split script:** `data/split_dataset.py`

### Why Temporal Split?
- Random split leaks future info (campaign spikes, seasonal effects, payer rate decay over time)
- LTV models must generalize **forward in time** — temporal holdout is the only fair test
- Train includes launch (high-volume, diverse behaviors) + steady-state
- Eval covers an unseen later period with lower payer rate — harder, more realistic

### Split Summary

| File | Date Range | Rows | Size | Payer Rate | Avg LTV30 | Purpose |
|------|-----------|------|------|-----------|-----------|---------|
| `cfm_train.csv` | Dec 16 – Jan 5 | 1,655,607 | 321 MB | 6.28% | 24,900 | **Model training.** Also works as local demo (~300MB). |
| `cfm_eval.csv` | Jan 6 – Jan 22 | 449,729 | 83 MB | 2.48% | 11,982 | **Temporal holdout.** Fair out-of-time evaluation. Prod demo candidate. |
| `cfm_demo.csv` | Dec 16 – Jan 22 (stratified sample) | 499,999 | 96 MB | 5.47% | 22,561 | **Production demo.** Stratified sample preserving payer distribution across all dates. ~100MB. |
| `cfm_predict.csv` | Jan 23 – Feb 21 | 518,713 | 90 MB | 1.47%* | 4,954* | **Inference/scoring.** Immature cohorts — LTV30 is unreliable. Use for prediction demo. |

*\*Predict set payer rate and LTV are artificially low because many users haven't reached 30-day maturity yet.*

### Key Properties

- **Train ∩ Eval = 0 rows** — zero temporal overlap, no data leakage
- **Train + Eval + Predict = 2,624,049 rows** — full dataset coverage, no gaps
- **Demo is a stratified subsample** of all mature data (Train + Eval period) — preserves payer rate distribution proportionally
- **Payer rate decreases over time** (6.28% → 2.48%) — expected as launch-week high-intent users taper off. The eval set tests whether the model generalizes to this harder distribution.

### Recommended Usage

| Use Case | File(s) |
|----------|---------|
| Train a model | `cfm_train.csv` |
| Evaluate model fairly | `cfm_eval.csv` |
| Quick local iteration / demo | `cfm_train.csv` (321MB) |
| Production app demo | `cfm_demo.csv` (96MB) or `cfm_eval.csv` (83MB) |
| Show inference on "new" users | `cfm_predict.csv` |
| Full retraining before deploy | `cfm_train.csv` + `cfm_eval.csv` (combined 404MB, all mature data) |
