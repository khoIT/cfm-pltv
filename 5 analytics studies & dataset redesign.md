A. Run 5 analytical studies from the existing pLTV dataset (cfm_pltv.csv)
focusing on topics:
  2. Temporal Analysis
  3. Cohort Comparison
  4. Causal Inference
  5. Seed Optimization Strategy
  6. Real‑Time Scoring (early prediction models)

For each topic:

1. Create a dedicated markdown report in /reports:
   e.g., reports/Temporal Analysis.md, Cohort Comparison.md …
   Each report must include:
   - Title and business context
   - Data selection SQL (Trino/Iceberg syntax)
   - Analytical steps (pandas or SQL)
   - Key charts (save under reports/plots/)
   - Findings with quantitative evidence
   - Business impact and next actions

2. Produce Python/Shell code blocks that can run end‑to‑end:
   - Pull appropriate data subset from cfm_pltv.csv
   - Validate sample size and data quality
   - Perform the analysis
   - Render Plotly figures
   - Write the markdown and save charts

3. After finishing all .md reports, choose **one** topic
   that yields the clearest business insight
   (likely “Temporal Analysis” or “Seed Optimization”)
   and build an interactive Streamlit page:
     webapp/pages/3c_<Chosen_Topic>.py
   Requirements:
   - Same polish as 3b_Late_Payer_Analysis.py
   - Add to sidebar navigation
   - Display KPI summary, interactive charts, and insights
   - Have module to allow users to choose a different dataset and the analysis will be rerun based on the .md content

Finally:
- Append a high‑level summary of all 5 analyses to reports/Synthesis_Summary.md

B. Redesign DataSet Experience of the app:
   - For Data Upload page:
      - When user upload a file, they can choose to split it into 1 or 2 datasets of their liking
      - Each dataset then are persisted in the app and can be retrieved across all pages when needed, either to train a model or rerun an analysis
   - Replace training data on the left sidebar with a feature called this Dataset Registry
      - All the datasets can be visible in the sidebar: which one is being chosen, which one can be selected.
      - When user switch the page, the previously chosen dataset for the page should be automatically highlighted too
      - For example, users may be on data_1 100k rows for Features and Models page, but when switch to Late Payer Analysis, they may prefer data_2 200k rows
      - Treat each page as a separate application and data registry must remember which application is using which dataset
      - When a dataset is delete, all pages that using it should display a warning prompt users to choose a new dataset