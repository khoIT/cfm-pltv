# Layer 1 — Decision Definition

*CFM pLTV / UA Seed Optimization*

## Business Decision

**Objective:** Predict player Lifetime Value at 30 days (pLTV30) to optimize User Acquisition seed lists for lookalike expansion on ad networks.

**When is the decision made?**  
At D7 after install — once enough behavioral signal has accumulated.

**Allowed Actions:**  
- Rank users by predicted LTV30 → select Top-K as high-value seeds  
- Feed seeds to ad networks (Facebook, Google, TikTok) for lookalike targeting  
- Adjust UA bids based on predicted value tiers  

**Success Metric:**  
- Higher ROAS from lookalike campaigns seeded with model-ranked users vs. heuristic (e.g., D7 revenue only)

---

## Key Performance Indicators

| KPI | Definition | Sample Value |
|-----|-----------|--------------|
| **ARPU (D30)** | Average Revenue Per User at 30 days | $0.42 |
| **Paying Rate (D30)** | % of users with LTV30 > 0 | 12.3% |
| **D7→D30 Multiplier** | Median ratio of LTV30 / rev_d7 among payers | 2.8× |
| **Median LTV30 (payers)** | Median spend among paying users | $3.15 |
| **Top-1% Revenue Share** | % of total revenue from top 1% users | 38% |

---

## Decision Blueprint

```
┌─────────────────────┐
│  Install + D0-D7    │
│  Behavioral Window  │
├─────────────────────┤
│  Feature Extraction │ ← Login, Gameplay, Payment, UA
├─────────────────────┤
│  pLTV30 Model       │ ← GBM / XGBoost
├─────────────────────┤
│  Rank & Segment     │ ← Top-K selection
├─────────────────────┤
│  UA Seed Export     │ → Ad Networks
└─────────────────────┘
```

## Data Window

- **Product:** CFM (CrossFire Mobile) — SEA launch 2025-12-16  
- **Feature window:** D0–D7 post-install  
- **Label window:** D0–D30 post-install  
- **Unique ID:** `vopenid`  
- **Gameplay ID mapping:** `vopenid → roleid` via `etl_new_register`
