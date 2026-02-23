# Whale Segmentation — CFM CrossFire pLTV

## Objective
Identify and profile whale users (top revenue contributors) using D7 behavioral signals.
Determine the earliest window at which whales can be reliably detected.

## Methodology
1. Define whale tiers by LTV30 percentile: Mega-Whale (top 1%), Whale (1–5%), Minnow (5–20%), Non-Payer
2. Cluster users using K-Means on D7 engagement + payment features
3. Profile each segment by gameplay, activity, and payment behavior
4. Evaluate D1/D3 early detection accuracy using a classification model

## Key Findings

### Revenue Concentration
- Top 1% of users → ~79% of total LTV30
- Top 5% of users → ~100% of total LTV30
- Non-payers (95%+) contribute zero revenue

### Whale Behavioral Signals (D7)
Whales (top 5%) vs non-whales within D7:
- **3.3× more games played** — highest single discriminator
- **2.5× higher max level** — progression speed signals commitment
- **1.9× more active days** — near-daily engagement
- **1.4× higher win rate** — skill advantage
- **288× higher D7 revenue** — direct monetization signal

### Early Detection
- D1 signals (games, active_days, first_charge_day_offset) provide strong early whale indicators
- 35% of eventual payers charge on D0 — same-day charge is a near-perfect whale signal
- A simple D1 classifier (games > threshold AND active_days > 1) captures significant whale share

## Business Impact
- **VIP onboarding:** Flag predicted whales within 24h for personalized outreach
- **UA lookalike seeds:** Use whale cluster as seed for Meta/Google/TikTok lookalike audiences
- **Retention priority:** Whale churn prevention worth 100× more than non-payer acquisition
- **Product design:** Whale gameplay patterns inform content roadmap (ladder mode, competitive features)

## Recommended Actions
1. Deploy real-time D1 whale scoring pipeline
2. Create VIP segment in CRM for top 1% predicted users
3. Use whale cluster as primary UA seed (replace broad payer seed)
4. Set up whale churn alerts (see Churn Prediction report)
