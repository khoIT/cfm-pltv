# Temporal Analysis — Full Dataset (2.6M users, Dec 16 → Feb 21)

## Dataset
- **2,624,049 users** across **68 install dates** (2025-12-16 → 2026-02-21)
- **Total D30 revenue: ₫49.2B** | **D7 revenue: ₫14.4B**
- Source file: `cfm_pltv_2025_12_16.csv`

## Business Context
Understanding how user quality, monetization, and engagement evolve from launch day through 2 months of steady-state UA for CFM Vietnam.

**Key Questions:**
- Does user quality degrade as launch hype fades?
- When is the optimal window for UA investment?
- Which channels and days deliver the best cohort quality?

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
3. Track engagement metrics (games, active days, K/D) for quality degradation signals
4. Run linear regression on each metric over time to detect statistically significant trends
5. Segment by media source and day-of-week to identify structural quality drivers

## Key Charts

### 1. Daily Install Volume & Payer Rates
![Volume & Payer Rates](plots/temporal_daily_volume_payer_rates.png)

### 2. ARPU Trends (D7 vs D30)
![ARPU Trends](plots/temporal_arpu_trend.png)

### 3. D7 Revenue as % of D30
![D7/D30 Ratio](plots/temporal_d7_d30_ratio.png)

### 4. Engagement Trends
![Engagement](plots/temporal_engagement_trends.png)

---

## Findings

### 1. Massive Launch-Day Spike, Rapid Normalization
- **Launch day (Dec 16):** 435,107 installs — 3–4× higher than subsequent days
- **Steady-state (Jan onwards):** 8,000–60,000 installs/day depending on paid UA bursts
- Organic installs dominate launch day; paid UA ramps up later

### 2. ₫34.8B (70.7%) of All Revenue Arrives After Day 7
- **Total D30 revenue: ₫49.2B** | **D7 revenue: ₫14.4B** | **D8–D30 revenue: ₫34.8B**
- D7 captures only **~29% of D30 revenue** on average
- This means ROAS calculations based on D7 are **~3.4× understated**

### 3. Late Payer Revenue = ₫14.7B (30% of Total) — Clarification
These two numbers are **not contradictory** — they measure different things:

| Metric | Amount | What it is |
|--------|--------|------------|
| ₫34.8B (70.7%) | All D8–D30 revenue | Revenue earned after Day 7 by **anyone** — including D7 payers who kept spending |
| ₫14.7B (30.0%) | Late payer LTV | Total D30 LTV of users who paid **₫0 in D7** but converted by D30 |

The ₫34.8B breaks down as:
- **~₫20.1B** — D7 payers who *continued* spending in D8–D30
- **~₫14.7B** — Late payers who *started* spending in D8–D30

Both are invisible at D7, but for different reasons:
- **Continued spenders:** D7 ROAS understates their full value by ~3.4×
- **Late payers:** The model assigns them near-zero predicted LTV because `rev_d7` has 95.8% feature importance — it cannot see users who haven't paid yet

### 4. All Quality Metrics Trending DOWN Over Time (Statistically Significant)

| Metric | Trend | R² | p-value | Meaning |
|--------|-------|----|---------|---------|
| ARPU D30 | 📉 −₫326/day | 0.64 | <0.0001 | Each cohort is worth less |
| D30 Payer Rate | 📉 −0.07%/day | 0.63 | <0.0001 | Fewer users convert |
| Late Payer Rate | 📉 −0.04%/day | 0.65 | <0.0001 | Late conversion also shrinking |
| Avg Games D7 | 📉 −0.27/day | 0.71 | <0.0001 | Users engage less |
| D7/D30 Ratio | 📈 +0.95%/day | 0.53 | <0.0001 | Late revenue window shrinking |

Launch hype is over. The Dec 16 cohort had the highest quality; by Feb, ARPU has declined substantially. Paid UA is replacing organic — later cohorts are increasingly paid (lower quality, not free).

### 5. Whale Concentration: 1% of Users = 80.5% of Revenue
- **Whale threshold (P99): ₫323,000**
- **26,333 whales** (1.00% of users) generate **₫39.6B** of the ₫49.2B total
- Whale ARPU: ₫1,503,211 vs overall ARPU ₫18,743 — **80× higher**
- **Late payers supply 34.6% of all whales** (9,103 of 26,333)

### 6. Late Payer Economics
- **53,032 late payers** (2.02% of all users)
- Average LTV: ₫277,850 (vs overall ARPU ₫18,743 — **15× higher**)
- Total revenue: ₫14.7B = **30% of all revenue**
- 9,103 are whales = **34.6% of all whales**
- Any seed list excluding late payers misses ~35% of whale signal sent to ad networks

### 7. Channel Economics — Wide Variance

| Channel | Users | ARPU D30 | Payer% | Late Payer% |
|---------|-------|----------|--------|-------------|
| **Apple Search Ads** | 265K | ₫35,987 | 6.75% | 3.05% |
| **Facebook Ads** | 134K | ₫25,336 | 7.40% | 3.20% |
| **Organic** | 1.02M | ₫24,842 | 5.28% | 2.33% |
| **TikTok** | 336K | ₫14,191 | 5.24% | 2.18% |
| **Google Ads** | 840K | ₫6,257 | 2.64% | 1.08% |

Google Ads delivers 840K users at only ₫6.3K ARPU — 5.7× lower than ASA. Likely broad targeting bringing low-intent installs.

### 8. Day-of-Week Effect: Wed/Thu Cohorts Monetize Best

| Day | Avg ARPU D30 | Payer% |
|-----|-------------|--------|
| Wednesday | ₫14,769 | 3.28% |
| Thursday | ₫13,652 | 2.98% |
| Sunday | ₫12,405 | 3.05% |
| Saturday | ₫12,332 | 2.95% |
| Friday | ₫10,827 | 2.75% |

---

## Business Impact & Recommended Actions

### Immediate (Week 1–2)
1. **Fix ROAS calculations** — multiply D7 ROAS by ~3.4× to estimate true D30 ROAS before making kill decisions
2. **Deploy enriched seed lists** — include ML-predicted late payers to capture the missing 35% of whale signal
3. **Pause or tighten Google Ads targeting** — ₫6.3K ARPU vs ₫36K for ASA; reallocate budget

### Short-Term (Month 1)
4. **Deploy D3 scoring pipeline** — stop waiting 7 days; D3 model retains ~97% of D7 accuracy
5. **Set cohort quality alerts** — ARPU is declining −₫326/day; automate alerts when a cohort drops below CPI breakeven
6. **Concentrate UA spend on Wed/Thu** — marginal ARPU lift from weekday scheduling

### Medium-Term (Month 2–3)
7. **Build whale retention system** — 1% of users = 80% of revenue; losing whales is existential risk
8. **Per-channel seed lists** — Facebook and ASA have 3× the late payer rate of Google; different seeds per network
9. **Multi-window ensemble scoring** — D1 triage → D3 primary → D7 final for fastest possible optimization loop
10. **Investigate quality decline root cause** — is it channel mix shift, creative fatigue, or audience saturation?
