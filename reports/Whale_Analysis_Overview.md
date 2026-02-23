# Whale Analysis Overview — CFM CrossFire pLTV

## Executive Summary

CrossFire Vietnam exhibits **extreme revenue concentration** characteristic of hardcore FPS games.
Analysis of 50,000 sampled users (Dec 16 2025 – Feb 18 2026) reveals:

| Metric | Value |
|---|---|
| Payer rate (D30) | **4.9%** |
| Top 1% users → % of revenue | **79.1%** |
| Top 5% users → % of revenue | **100%** |
| Payer LTV30 mean | ₫383,611 |
| Payer LTV30 median | ₫129,000 |
| Whale LTV30 (top 5%) mean | ₫116,648 |
| Non-whale payer LTV30 mean | ₫405 |

**Implication:** The entire revenue base is driven by ~1% of users. Standard payer-rate optimization
is insufficient — the business must identify, retain, and expand the whale segment specifically.

---

## Revenue Concentration

The Pareto principle is extreme here — it's not 80/20, it's **79/1**:

- **Top 1%** of all users → 79.1% of total LTV30
- **Top 5%** of all users → 100% of total LTV30 (non-top-5% users contribute nothing net)
- Non-payers (95.1%) contribute ₫0

This means every analysis, model, and UA decision should be evaluated through the lens of
**whale capture rate**, not just payer rate or average ARPU.

---

## Whale Behavioral Profile (D7 data, top 5% vs rest)

| Signal | Non-Whale | Whale (top 5%) | Ratio |
|---|---|---|---|
| Games played (D7) | 25.5 | 84.9 | **3.3×** |
| Win rate | 32.2% | 44.6% | 1.4× |
| K/D ratio | 1.67 | 1.98 | 1.2× |
| Avg score | 5,748 | 7,452 | 1.3× |
| Max level seen | 22 | 54 | **2.5×** |
| Active days (D7) | 3.2 | 6.2 | **1.9×** |
| Revenue D7 | ₫405 | ₫116,648 | **288×** |

**Key insight:** Whales are not just richer — they play **3.3× more games**, reach **2.5× higher levels**,
and are active **1.9× more days** within the first week. These behavioral signals are detectable early.

---

## First Purchase Timing

Among payers (n=1,441 in sample):

| Window | Count | % of payers |
|---|---|---|
| D0 (same-day charge) | 501 | **34.8%** |
| D1–D3 | 622 | **43.2%** |
| D4–D7 | 318 | **22.1%** |

**78% of eventual payers have charged by D3.** This creates a strong signal for early whale detection
and justifies D1/D3 prediction windows for real-time scoring.

---

## Acquisition Channels

| Media Source | Users (sample) |
|---|---|
| Organic | 19,261 (38.5%) |
| Google Ads | 15,645 (31.3%) |
| TikTok | 6,612 (13.2%) |
| Apple Search Ads | 5,228 (10.5%) |
| Facebook Ads | 2,702 (5.4%) |

All users are Vietnam-only (`first_country_code = VN`). Channel quality varies significantly —
whale rate per channel is unknown without further analysis and is a key open question.

---

## Skill Distribution

| Metric | Mean | Median | Max |
|---|---|---|---|
| K/D ratio | 1.51 | 1.05 | 243 |
| Win rate | 26.7% | 30.0% | 100% |

High variance in skill suggests a mixed player base (casual + competitive). The correlation
between skill and spending is a key research question — do skilled players spend more (intrinsic motivation)
or do spenders buy skill (pay-to-win)?

---

## AI-Generated Analysis Roadmap

Based on the above findings, five targeted analyses have been implemented:

| # | Analysis | Priority | Key Question |
|---|---|---|---|
| 1 | **Whale Segmentation** | 🔴 Critical | Who are the whales and can we identify them at D1? |
| 2 | **Time-to-First-Purchase** | 🔴 High | When do whales convert and what predicts faster conversion? |
| 3 | **Channel × Whale Quality** | 🟠 High | Which UA channels deliver the highest whale rate? |
| 4 | **Churn Prediction (Payers)** | 🔴 Critical | Which payers will churn before D30? |
| 5 | **Skill-to-Spend Correlation** | 🟠 Medium | Does skill drive spending or does spending buy skill? |

---

## Recommended Actions

1. **Immediate:** Deploy D3 whale scoring — flag top 1% predicted users for VIP treatment within 72h of install
2. **UA Optimization:** Shift budget toward channels with highest whale rate (not just lowest CPI)
3. **Retention:** Build payer churn early-warning system — a churned whale is a massive revenue loss
4. **Product:** Investigate skill-spend relationship to inform monetization design
5. **Lookalike Seeds:** Use confirmed whales (not just payers) as UA lookalike seeds

---

*Generated: Feb 20 2026 | Data: Dec 16 2025 – Feb 18 2026 | Sample: 50,000 users*
