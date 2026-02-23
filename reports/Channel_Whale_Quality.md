# Channel × Whale Quality — CFM CrossFire pLTV

## Objective
Evaluate UA acquisition channels not just by volume or CPI, but by **whale rate** —
the fraction of acquired users who become top-1% revenue contributors.

## Methodology
1. Group users by `media_source`
2. Compute per-channel: payer rate, whale rate (top 1% LTV), ARPU, total LTV30
3. Rank channels by whale rate and revenue efficiency
4. Compare D7 behavioral quality across channels (games, active_days, win_rate)

## Key Findings

### Channel Mix (sample)
| Channel | Users | Share |
|---|---|---|
| Organic | ~38.5% | Baseline quality benchmark |
| Google Ads | ~31.3% | Largest paid channel |
| TikTok | ~13.2% | Fast-growing, younger demographic |
| Apple Search Ads | ~10.5% | iOS-only, high intent |
| Facebook Ads | ~5.4% | Retargeting-heavy |

### Channel Quality Dimensions
- **Whale rate:** % of channel users in top 1% LTV30
- **ARPU D30:** Average revenue per user at 30 days
- **Payer rate:** % who make any purchase
- **Engagement quality:** avg games, active days, win rate at D7

### Organic as Benchmark
Organic users represent self-selected high-intent players — their whale rate sets the ceiling
for paid channel quality. Channels approaching organic whale rate deliver superior ROI.

## Business Impact
- **Budget reallocation:** Shift spend from low-whale-rate to high-whale-rate channels
- **Bid strategy:** Use whale rate (not just ROAS) as the primary campaign optimization signal
- **Creative targeting:** Channels with high engagement but low whale rate may need creative refresh
- **Lookalike quality:** Build channel-specific lookalike seeds from each channel's whale users

## Recommended Actions
1. Report whale rate alongside CPI and ROAS in all campaign dashboards
2. Set minimum whale rate threshold (e.g. 0.8% = 80% of organic) for channel investment
3. Pause campaigns on channels with whale rate < 0.3% regardless of volume
4. Build channel-specific pLTV models if behavioral profiles differ significantly
