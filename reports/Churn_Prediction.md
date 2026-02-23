# Churn Prediction (Payers) — CFM CrossFire pLTV

## Objective
Predict which paying users will stop spending before D30, enabling proactive retention interventions.
For a whale-intensive game, a single churned mega-whale represents enormous revenue loss.

## Methodology
1. Define churn: payer at D7 (rev_d7 > 0) who does NOT appear in top LTV30 tier
2. Build binary classifier: "will this D7 payer reach their LTV30 potential?"
3. Features: D7 gameplay trajectory, payment frequency, engagement depth
4. Evaluate: precision/recall on payer churn, with emphasis on whale churn recall

## Key Findings

### Payer Segmentation
- **One-time payers:** Single transaction in D7, no repeat — high churn risk
- **Repeat payers:** Multiple transactions in D7 — strong retention signal
- **Whales:** High txn_cnt_d7 + high rev_d7 — retention is critical priority

### Churn Risk Signals
- Low `active_days_d7` despite having paid — disengagement after purchase
- Declining `games` count relative to D1 baseline
- Single transaction with low `rev_d7` — trial purchase, not committed
- High `first_charge_day_offset` (D5–D7 first charge) — late converters churn faster

### Retention Indicators
- `txn_cnt_d7 >= 2` — repeat purchase within D7 is strongest retention signal
- `active_days_d7 >= 5` — near-daily engagement predicts continued spending
- `max_level_seen` progression — players advancing in content stay longer
- Low `first_charge_day_offset` (D0–D1) — early committed payers have higher LTV30

## Business Impact
- **Revenue protection:** Identifying at-risk whales 2–3 weeks before churn enables intervention
- **Retention ROI:** Cost of retention offer << cost of re-acquiring a whale via paid UA
- **LTV accuracy:** Churn prediction improves pLTV model accuracy for mid-funnel users
- **Product signals:** Churn patterns reveal content gaps or monetization friction points

## Recommended Actions
1. Deploy D7 churn risk score for all payers — flag high-risk users for CRM outreach
2. Trigger retention offer for payers with churn_risk > 0.7 and rev_d7 > ₫50,000
3. A/B test retention interventions: exclusive content vs discount offer vs social feature
4. Feed churn predictions back into pLTV model as a feature (Feedback & Learning loop)
