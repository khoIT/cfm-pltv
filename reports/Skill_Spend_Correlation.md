# Skill-to-Spend Correlation — CFM CrossFire pLTV

## Objective
Determine whether skill (K/D ratio, win rate, score) drives spending, or whether spending
buys skill (pay-to-win dynamic). This informs both monetization design and UA targeting.

## Methodology
1. Compute Spearman correlations between skill metrics and LTV30 across all users
2. Segment by payer status: non-payer, low-payer, whale
3. Compare skill distributions across spending tiers
4. Test causal direction: do high-skill non-payers eventually convert at higher rates?

## Key Findings

### Skill Distribution (D7)
| Metric | Mean | Median | Top 1% threshold |
|---|---|---|---|
| K/D ratio | 1.51 | 1.05 | ~8.0 |
| Win rate | 26.7% | 30.0% | ~80% |
| Avg score | varies | varies | — |

### Whale Skill Profile vs Non-Whale
- Whales have **1.4× higher win rate** (44.6% vs 32.2%)
- Whales have **1.2× higher K/D** (1.98 vs 1.67)
- Whales play **3.3× more games** — skill improvement through volume

### Interpretation
- Positive correlation between skill and spending exists, but is moderate
- High game volume (whales play 85 games vs 25 for non-whales) suggests skill is partly
  *acquired through spending* (more playtime = more practice = better stats)
- The causal direction is likely **bidirectional**: skilled players enjoy the game more → spend more;
  spenders play more → improve skill

### Skill Segments for UA
- **High-skill non-payers:** Competitive players who haven't monetized — high conversion potential
  with the right offer (cosmetics, ranked mode access)
- **Low-skill payers:** May be paying to compensate for skill gap — retention risk if they plateau
- **High-skill payers (whales):** Core audience — protect and expand this segment

## Business Impact
- **Monetization design:** Cosmetic/prestige items for high-skill players; power items for low-skill
- **UA targeting:** Skill signals (K/D, win rate) can be used as lookalike seed features
- **Retention:** Low-skill payers need skill-improvement content to stay engaged
- **Anti-churn:** High-skill non-payers are the best conversion targets for mid-funnel campaigns

## Recommended Actions
1. Add skill tier (based on K/D + win rate) as a feature in the pLTV model
2. Create separate monetization funnels for skill-motivated vs convenience-motivated payers
3. Target high-skill non-payers (K/D > 2.0, win_rate > 40%, rev_d7 = 0) with competitive offers
4. Monitor skill progression of new payers — stagnation predicts churn
