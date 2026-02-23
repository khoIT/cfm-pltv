# Time-to-First-Purchase Analysis — CFM CrossFire pLTV

## Objective
Understand when users make their first purchase and whether faster conversion predicts higher LTV30.
Identify the optimal intervention window for monetization nudges.

## Methodology
1. Compute survival curve: % of eventual payers who have charged by each day D0–D7
2. Segment payers by first-charge timing: D0, D1–D3, D4–D7
3. Compare LTV30 across timing segments — does faster = higher value?
4. Identify behavioral signals that predict same-day (D0) conversion

## Key Findings

### Conversion Timing (payers only)
| Window | % of Payers | Cumulative |
|---|---|---|
| D0 (same day) | ~35% | 35% |
| D1–D3 | ~43% | 78% |
| D4–D7 | ~22% | 100% |

**78% of all eventual payers have converted by D3.** This validates D3 as the primary scoring window.

### Does Faster = Higher Value?
- D0 payers tend to be the highest LTV segment — they arrive with purchase intent
- D4–D7 payers are often "nudged" converters — lower average LTV30
- The gap between D0 and D4–D7 payer LTV is a key business metric

### Behavioral Predictors of D0 Conversion
- High `games_d7` in first session
- `max_level_seen` progression within hours of install
- Login channel (direct vs referral)
- Device type and OS

## Business Impact
- **Offer timing:** Show first purchase offer within first session for high-engagement users
- **Push notifications:** D2–D3 nudge for users who haven't converted but show high engagement
- **Campaign optimization:** Bid higher for users predicted to convert on D0 (highest LTV)
- **Payback period:** D0 converters recover UA cost faster — prioritize in ROAS calculations

## Recommended Actions
1. Implement session-level trigger: show offer when `games_played >= 3` in first session
2. Set D3 re-engagement push for non-converted high-engagement users
3. Exclude D4–D7 converters from lookalike seeds (lower quality signal)
4. Track D0 conversion rate as a primary UA campaign quality metric
