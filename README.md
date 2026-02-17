# CFM Decision Intelligence Demo

**Decision-Centric Intelligence Framework** applied to the CFM (CrossFire Mobile) player-level dataset for pLTV / UA Seed Optimization.

## Overview

This project demonstrates how each layer of the Decision-Centric Intelligence Loop operates, from decision definition through causal feedback, using real CFM player telemetry data (~350 MB).

## Directory Structure

```
cfm_pltv/
 ├── data/                            # Raw + derived feature tables
 │   ├── cfm_pltv.csv                 # Real dataset (350 MB, not in repo)
 │   └── cfm_pltv_sample.csv          # Synthetic sample (30k rows) for demo
 ├── notebooks/                       # Analysis notebooks per Loop layer
 │   ├── 1_decision_definition.ipynb
 │   ├── 2_feature_exploration.ipynb
 │   ├── 3_modeling_eval.ipynb
 │   ├── 4_action_simulation.ipynb
 │   └── 5_feedback_learning.ipynb
 ├── reports/                         # Auto-generated .md artifacts
 │   ├── decision_definition.md
 │   ├── feature_store_overview.md
 │   ├── model_training.md
 │   ├── evaluation_metrics.md
 │   ├── action_simulation.md
 │   ├── feedback_stub.md
 │   └── CFM_Decision_Intelligence_Summary.md
 ├── reports/plots/                   # PNG charts embedded in reports
 ├── webapp/                          # Interactive Streamlit demo
 │   ├── app.py                       # Entry point
 │   ├── pages/
 │   │   ├── 1_Decision_Definition.py
 │   │   ├── 2_Features_and_Model.py
 │   │   ├── 3_Evaluation_and_Insights.py
 │   │   ├── 4_Action_and_Simulation.py
 │   │   └── 5_Feedback_and_Learning.py
 │   └── requirements.txt
 ├── utils/
 │   ├── reporting.py                 # Helper to render markdown w/ charts
 │   └── synthetic_data.py            # Generates demo data
 ├── decision-framework.md            # Reference: Decision Loop + Eval tables
 └── README.md
```

## Framework → Artifact Mapping

| # | Framework Layer | Notebook | Report | Webapp Page |
|---|----------------|----------|--------|-------------|
| 1 | Decision Definition | `1_decision_definition.ipynb` | `decision_definition.md` | Decision Definition |
| 2 | Feature / ML Ops | `2_feature_exploration.ipynb` | `feature_store_overview.md` | Features & Model |
| 3 | Models + Evaluation | `3_modeling_eval.ipynb` | `model_training.md` / `evaluation_metrics.md` | Evaluation & Insights |
| 4 | Action Simulation | `4_action_simulation.ipynb` | `action_simulation.md` | Action & Simulation |
| 5 | Causal Feedback | `5_feedback_learning.ipynb` | `feedback_stub.md` | Feedback & Learning |

## Quick Start

### 1. Install Dependencies
```bash
cd cfm_pltv
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r webapp/requirements.txt
```

### 2. Generate Sample Data (if no real data)
```bash
python -m utils.synthetic_data
```

### 3. Run Streamlit App
```bash
streamlit run webapp/app.py
```

### 4. Run Notebooks
Open notebooks in Jupyter / VS Code and run cells sequentially.

## Data

- **Real data:** Place `cfm_pltv.csv` (from SQL query) in `data/`
- **Sample data:** Auto-generated `cfm_pltv_sample.csv` (30k synthetic rows) for demo
- The app auto-detects: uses real data if available, falls back to sample

## Tech Stack

- **Python 3.10+**, pandas, numpy, scikit-learn, xgboost
- **Visualization:** Plotly (interactive), Matplotlib (static for reports)
- **Web App:** Streamlit
- **Reports:** Markdown with embedded PNG charts
