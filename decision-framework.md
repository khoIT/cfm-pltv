## Decision-Centric Intelligence Loop
To really turns raw telemetry into decision-ready scores—Value (pLTV), Risk (churn/survival), Responsiveness (offer sensitivity), and Intent (player role)—applied across the right entity layers (user, cohort, campaign), we needs feature store and MLOps platform.

To really make data actionable for every important decision, we need a framework that combines:
- Decision Definition
- Feature Store 
- ML Ops Platform (model training and back-testing)
- A/B testing framework

# Decision-Centric Intelligence Loop

| Layer | Stage | Purpose | Key Questions Answered | Example Outputs |
|-------|--------|----------|------------------------|------------------|
|  | **Decision Definition** | Define the business decision to optimize and align DS + business | 🎯 What objective are we optimizing? <br> ⏱ When is the decision made? <br> 🎛 What actions are allowed? <br> 📏 How do we measure success? | • Decision blueprint <br> • Optimization goal <br> • Treatment design rules |
| **Feature / ML Ops Platform** | Raw Logs | Collect reliable behavioral & monetization signals | Do we have complete and trustworthy signals? | • Gameplay logs <br> • Payment logs <br> • UA cost logs |
|  | Features | Transform logs into reusable behavioral & economic features | What signals explain user value or risk? | • Feature Store tables <br> • Aggregated user state snapshots |
|  | Models | Predict user outcomes or latent intent | Who is likely to churn? <br> Who has high LTV potential? <br> Who is price sensitive? | • pLTV scores <br> • Churn probabilities <br> • Intent embeddings |
| **Insights / Signals Layer** | Decisioning | Convert model outputs into actionable cohorts & rules | Which users should be treated? <br> When should treatment trigger? | • Ranked segments <br> • Trigger thresholds <br> • Decision policies |
| **Execution Layer (UA / LiveOps)** | Actions | Apply treatments to targeted cohorts | What action is executed? | • Push campaigns <br> • Discounts <br> • Bid adjustments <br> • Personalization |
| **Causal Learning Layer** | Experimentation & Feedback | Measure treatment effectiveness and continuously optimize decisions | Did treatment cause improvement? <br> Which action works best for which cohort? | • Uplift measurements <br> • A/B test results <br> • Treatment policy updates |

## pLTV vs Churn Evaluation Framework
# Evaluation Framework: pLTV vs Churn

| Layer | Evaluation Aspect | pLTV (Revenue Uplift Decision) | Churn (Churn Prevention Decision) |
|-------|-------------------|--------------------------------|-----------------------------------|
| **Business Objective** |  | Ensures ranking aligns with maximizing future revenue concentration. | Ensures risk scoring aligns with reducing future churn losses. |
| **Offline - Signal Concentration** | Lift Curve | Shows whether most of the money is concentrated in the top-ranked users. | Shows whether most churn risk is concentrated in the top-ranked users. |
|  | Precision@K | Among the top K users selected, how many actually turn out to be high spenders. | Among the top K risky users selected, how many actually churn. |
|  | Recall@K | How many of all high spenders are included in the top K; guides how aggressive Top-K targeting should be. | How many of all churners are included in the top K. |
| **Offline - Ranking / Classification Quality** | Spearman Rank Correlation | Checks if the model orders users from low to high value correctly across everyone. | Evaluates risk ordering consistency; secondary but useful for stability checks. |
|  | ROC / AUC | Not very important here since revenue is continuous, not yes/no. | Measures ability to distinguish churners vs non-churners; core classification metric. |
|  | Calibration | Ensures predicted revenue magnitude matches reality; prevents overbidding in UA. | Ensures predicted churn probability reflects true risk; prevents misallocation of retention budget. |
| **Online Impact Measurement** | Uplift vs Baseline | Verifies model adds incremental revenue vs heuristic (e.g., LTV7d); ensures real business gain. | Verifies intervention reduces churn beyond control group; ensures prediction translates to action. |
|  | Treatment Sensitivity | Tests how revenue outcome changes when used in ads activation; validates downstream impact. | Tests how retention changes after intervention; mandatory for proving model usefulness. |
|  | Economic Efficiency (ROI / ROAS) | Confirms predicted high value converts into profitable ad spend. | Confirms retention incentive cost produces positive incremental profit. |
| **Stability & Governance Layer** | Time Dynamics (Revenue / Retention Curve) | Tracks how revenue grows over time after targeting. | Tracks how long retention improvement lasts. |
|  | Robustness / Stability | Checks if model works across different user groups and time periods. | Ensures risk prediction holds across behavioral shifts and lifecycle stages. |