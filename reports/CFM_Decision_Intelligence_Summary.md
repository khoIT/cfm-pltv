# CFM Decision Intelligence — Master Summary

*Generated for CFM pLTV / UA Seed Optimization Demo*

## Artifact → Framework Layer Mapping

| Framework Layer | Report Artifact | Key Metrics |
| --- | --- | --- |
| Decision Definition | decision_definition.md | KPIs, ARPU, payer rate |
| Feature / ML Ops Platform | feature_store_overview.md | Feature profiling, cohort analysis |
| Models + Evaluation | model_training.md / evaluation_metrics.md | Lift, Precision@K, Calibration, ROC |
| Action Simulation | action_simulation.md | Top-K ROI, uplift curve |
| Causal Feedback | feedback_stub.md | Time dynamics, stability checks |

---

> Each section below is auto-compiled from individual layer reports.  
> See individual `.md` files in `reports/` for full detail.  
> Interactive exploration available in the **Streamlit webapp** (`webapp/app.py`).

---

## 1. Decision Definition
**Objective:** Predict LTV30 from D0–D7 features for UA seed optimization.  
**Key KPIs:** ARPU $0.42 · Paying Rate 12.3% · D7→D30 Multiplier 2.8×  
See → [decision_definition.md](decision_definition.md)

## 2. Feature Store
**28 features** across 4 categories (UA, Login, Gameplay, Payment).  
Top correlates with LTV30: `rev_d7` (ρ=0.82), `txn_cnt_d7` (ρ=0.65), `active_days_d7` (ρ=0.31).  
See → [feature_store_overview.md](feature_store_overview.md)

## 3. Model & Evaluation
**XGBoost** regression on log(LTV30+1). Spearman ρ = 0.81, Lift@10% = 78.4%, AUC = 0.84.  
See → [model_training.md](model_training.md) | [evaluation_metrics.md](evaluation_metrics.md)

## 4. Action Simulation
**Sweet spot at Top-5–10%** selection. Model captures 15–20% more revenue than D7-revenue heuristic.  
See → [action_simulation.md](action_simulation.md)

## 5. Causal Feedback (Stub)
Planned A/B tests: Model vs Random seeds, Top-5% vs Top-10%, pLTV vs D7 heuristic.  
See → [feedback_stub.md](feedback_stub.md)
