# Layer 3 — Evaluation & Insights

*Offline evaluation of the pLTV30 XGBoost model — Decision-Centric Evaluation Framework*

---

## Dataset Selection

The page supports two evaluation modes via a **Train / Test toggle**:

| Mode | Dataset | Description |
|------|---------|-------------|
| 🏋️ Train (in-sample) | `cfm_pltv_train` | 80% of mature users. Metrics may be optimistic — model was trained on this data. |
| 🧪 Test (holdout) | `cfm_pltv_test` | 20% holdout, never seen during training. Use this for unbiased real-world performance estimates. |

**Mature users** = installed ≥ 30 days before the data dump date, so LTV30 is fully realized.  
Always prefer **Test mode** when reporting model quality to stakeholders.

---

## Baseline Heuristics Comparison

The page allows toggling on one or more **single-feature baselines** to overlay against the XGBoost model on every chart. This answers: *does the ML model add value over a simple rule?*

| Baseline | Column Used | Rationale |
|----------|-------------|-----------|
| rev_d7 | `rev_d7` | D7 revenue — strongest single predictor |
| logins_d7 | `logins_d7` | Engagement proxy |
| sessions_d7 | `sessions_d7` | Session count |
| payer_d7 | `payer_d7` | Binary payer flag |

If `rev_d7` nearly matches XGBoost on Lift and Spearman, a simple revenue-based rule may suffice. If XGBoost is materially better, the model is learning non-obvious feature interactions.

---

## Signal Concentration — Lift Curve

The lift curve shows cumulative revenue captured when users are ranked by each strategy (model, baseline, or random).

Three lines are always shown:
- **Oracle (perfect model)** — green dashed line. Theoretical maximum: users sorted by actual `ltv30`. This is the ceiling no model can exceed.
- **XGBoost model** — blue solid line. The model's actual ranking performance.
- **Random** — diagonal. What you'd get with no model at all.

Vertical markers are drawn at 1%, 5%, 10%, and 20% to make it easy to read off values.

### Cumulative Revenue @ Key Thresholds

The scorecard table shows model vs oracle at each breakpoint:

| Top-K % | Oracle (max possible) | XGBoost | Gap to Oracle | Random |
|---------|-----------------------|---------|---------------|--------|
| Top 1%  | ~38–42%               | ~33–38% | ~3–5 pp       | 1%     |
| Top 5%  | ~65–70%               | ~60–66% | ~3–6 pp       | 5%     |
| Top 10% | ~80–84%               | ~76–81% | ~2–5 pp       | 10%    |
| Top 20% | ~91–94%               | ~88–92% | ~2–4 pp       | 20%    |

*Actual values depend on the active dataset and model. A small gap to oracle (< 5 pp) indicates near-optimal ranking.*

---

## Scorecard

The scorecard summarises all strategies side-by-side:

| Column | Definition |
|--------|-----------|
| Spearman ρ | Rank correlation between predicted and actual LTV30. Range −1 to 1; higher is better. |
| Lift@10% | % of total revenue captured by top 10% of users ranked by this strategy. |
| Oracle@10% | Theoretical maximum Lift@10% (sorted by actual LTV30). |
| Gap to Oracle | Lift@10% shortfall vs oracle in percentage points. Smaller = better. |
| Prec@5% | Precision at top 5%: fraction of selected users who are truly high-value (top 5% by actual LTV30). |
| Recall@10% | Recall at top 10%: fraction of all high-value users captured in the top 10% selection. |
| AUC | ROC-AUC for payer vs non-payer classification. |

---

## Precision@K & Recall@K

**High-value user definition: top 5% by actual LTV30** (95th percentile threshold).  
This is a stricter definition than top 10%, making the metrics more meaningful — a model must rank very precisely to score well here.

| Metric | Definition | Ideal value |
|--------|-----------|-------------|
| Precision@K | Of the top K% users selected, what fraction are truly high-value? | 1.0 at small K |
| Recall@K | Of all high-value users, what fraction appear in the top K%? | 1.0 at large K |

**Interpretation guide:**
- Precision@1% ≈ 50–80% → model is concentrating high-value users at the very top
- Recall@20% ≈ 80–95% → nearly all whales are captured in the top quintile
- If Precision@1% ≈ Recall@1% / 5 → consistent with a well-calibrated top-5% definition

---

## Ranking Quality — Spearman ρ

Measures monotonic agreement between predicted and actual LTV30 rank order.

| ρ range | Interpretation |
|---------|---------------|
| > 0.85 | Excellent — near-perfect rank ordering |
| 0.70–0.85 | Good — suitable for UA seed selection |
| 0.50–0.70 | Moderate — model adds value but misses some signal |
| < 0.50 | Weak — consider feature engineering or more data |

---

## Calibration Plot

Predicted LTV30 (x-axis) vs actual LTV30 (y-axis) across 10 equal-width prediction bins.  
A perfectly calibrated model lies on the diagonal. Systematic deviation indicates:
- **Curve above diagonal** → model under-predicts (conservative)
- **Curve below diagonal** → model over-predicts (optimistic)

Calibration matters for budget allocation: if the model over-predicts whale LTV by 2×, UA bids will be inflated.

---

## ROC / AUC — Payer Classification

Binary classification: payer (`ltv30 > 0`) vs non-payer.

| AUC | Interpretation |
|-----|---------------|
| > 0.90 | Excellent payer detection |
| 0.80–0.90 | Good — suitable for targeting |
| 0.70–0.80 | Moderate |
| < 0.70 | Weak |

All active baselines are overlaid on the ROC curve with their individual AUC scores for direct comparison.

---

## Summary vs Framework

| Eval Aspect | Metric | What to look for |
|------------|--------|-----------------|
| Signal Concentration | Lift@10% vs Oracle@10% | Gap < 5 pp = near-optimal ranking |
| Precision | Precision@5% | > 0.5 = model reliably finds whales |
| Recall | Recall@10% | > 0.7 = captures most high-value users |
| Ranking | Spearman ρ | > 0.75 = strong rank ordering |
| Classification | AUC | > 0.80 = good payer separation |
| Calibration | Plot shape | Close to diagonal = well-calibrated |
| Baseline comparison | Lift gap vs rev_d7 | Model should exceed rev_d7 baseline |

---

## Key Takeaway

If **XGBoost beats every single-feature baseline** on Lift, Precision, and Spearman, the model is learning non-obvious signal combinations that no single heuristic captures.

If a baseline like `rev_d7` is nearly as good, a simple rule may suffice — saving the complexity of maintaining a full ML pipeline.

Always evaluate on the **Test (holdout)** dataset before making production decisions. Train-set metrics are optimistic by construction.
