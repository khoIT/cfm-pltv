# CFM Data Chatbot — Current Implementation Summary

## 1. Architecture Overview

The chatbot is implemented across two files:

| File | Lines | Role |
|------|-------|------|
| `webapp/pages/CFM_Chatbot.py` | 939 | Main page: UI, pipeline, DuckDB, TF-IDF retrieval, pLTV inference, chart rendering |
| `webapp/utils/llm_providers.py` | 178 | Multi-provider LLM adapter (OpenAI, Anthropic, Google Gemini) |

### High-level flow

```
User question
  → classify_intent() — keyword match → "cfm" or "general"
  → retrieve_snippets() — TF-IDF over reports/*.md → top-3 report excerpts
  → call_llm() — system prompt + report context + conversation history + question
  → parse_llm_response() — extract SQL, CHARTS JSON array, and prose
  → execute_sql() — DuckDB over cfm_pltv_Feb22.csv (with safety validation)
  → render_artifact() — st.dataframe + ECharts + CSV export
  → If pLTV scoring intent detected → run_pltv_inference() separately
  → Show citations panel
```

---

## 2. Spec vs Implementation — Feature Checklist

### Fully Implemented (✅)

| Spec Requirement | Implementation |
|---|---|
| **DuckDB over CSV** — lazy query via `read_csv_auto` | `get_duckdb_conn()` creates a cached `cfm_features` view over `cfm_pltv_Feb22.csv` |
| **SQL safety check** — reject destructive keywords | `validate_sql()` blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc. (12 keywords, exceeding the spec's 11) |
| **Auto LIMIT** — append LIMIT if missing | `execute_sql()` appends `LIMIT 5000` if no LIMIT clause found |
| **SQL self-correction** — retry on error | On SQL error, sends error back to LLM for a second attempt with corrected SQL |
| **Report retrieval** — TF-IDF over `reports/*.md` | `build_report_index()` chunks by `##` headings, builds TF-IDF (5000 features), cosine similarity retrieval. Also indexes root-level `.md` files beyond `reports/` |
| **Citation format** — collapsible sources panel | `st.expander("📚 Sources used")` shows file, heading, relevance score, and snippet preview |
| **Multi-provider LLM** — OpenAI, Anthropic, Gemini | `llm_providers.py` implements streaming for all three. Models include latest (gpt-5, claude-opus-4, gemini-2.5-pro) |
| **System prompt** — CFM context, key facts, rules | Comprehensive 113-line system prompt with dataset facts, column schema, and response format instructions |
| **CFM/General mode classification** — keyword match | `classify_intent()` with 52 keywords covering pLTV, whales, channels, gameplay, etc. |
| **pLTV model inference** — load pickle, score users | `run_pltv_inference()` loads from `models/*/model.pkl`, handles missing columns, returns scored DataFrame |
| **Model selector** — sidebar dropdown | Lists available models from `models/` dir, with max-rows slider (10K–200K) |
| **Starter question pills** — sidebar clickable prompts | 8 pre-defined questions matching the spec |
| **Clear chat** — sidebar button | Clears `st.session_state.messages` and reruns |
| **Schema viewer** — sidebar expander | Shows full column schema + source file caption |
| **API key management** — per-provider, session-only | Stored in `st.session_state["api_keys"]`, password input, never written to disk |
| **Session state keys** — messages, chat_mode, api_keys | All specified keys are used |
| **Conversation history** — context window | Last 20 messages sent to LLM for multi-turn context |
| **CSV export** — download button per result | `st.download_button` for every SQL result |
| **Sidebar nav integration** — chatbot link in sidebar | `render_sidebar()` in `shared.py` has `st.sidebar.page_link("pages/CFM_Chatbot.py", ...)` |

### Partially Implemented (⚠️)

| Spec Requirement | Current State | Gap |
|---|---|---|
| **Multiple charts per response** | LLM is instructed to output `CHARTS: [...]` JSON array. `render_echarts()` renders each. Also has `auto_generate_charts()` fallback. | Chart generation depends on LLM correctly outputting the JSON array — no structured output enforcement. Auto-fallback only generates 1–3 basic charts based on column types. |
| **Chart types** — bar, line, scatter, pie, heatmap, radar, funnel, boxplot | bar, line, scatter, pie, funnel, heatmap, boxplot all have ECharts builders. | **Radar chart** type is listed in the prompt but has no `build_echarts_option()` handler — falls through to bar fallback. |
| **HTML chart export** | Not implemented. Only CSV export exists. | Spec calls for `⬇️ Export Chart HTML` and `⬇️ Export PNG` buttons. ECharts has `saveAsImage` in toolbox (built-in), but no explicit HTML/PNG download buttons. |
| **Canvas mode ("✏️ Edit chart")** | Not implemented. | Spec asks for a `st.text_area` with chart JSON for user editing + "Re-render" button. Not present. |
| **Deterministic sub-flows** — whale, channel, late-payer, pLTV | Only pLTV scoring has a dedicated sub-flow (`is_pltv_request` check). Whale, channel, and late-payer flows rely on the LLM generating appropriate SQL. | Spec envisions deterministic keyword-triggered sub-flows for whale analysis, channel comparison, and late-payer analysis with specific auto-computed charts. Currently these are all delegated to the LLM. |
| **Currency toggle** | System prompt mentions VND. `format_currency` and `convert_vnd` are imported but not used in chatbot responses. | SQL results display raw numbers. The chatbot doesn't apply the sidebar currency toggle to query results. |
| **pLTV scored table** | Top 1% shown with bar + pie charts. Full table downloadable. | Charts use `vopenid` as x-axis for bar chart (not meaningful for large datasets). Score distribution histogram (spec requirement) is not generated. |

### Not Implemented (❌)

| Spec Requirement | Notes |
|---|---|
| **Pareto live calculator** with dynamic percentile slider | Spec §14.1 — not present |
| **Late-payer opportunity quantifier** | Spec §14.2 — not a dedicated flow |
| **Cohort quality trend alert** with regression | Spec §14.3 — relies on LLM |
| **Feature importance explainer** from model metadata | Spec §14.4 — model metadata loaded but importance chart not rendered |
| **Channel ROI ranker** with CPI proxy | Spec §14.5 — not a dedicated flow |
| **Engagement → conversion funnel** dose-response | Spec §14.6 — relies on LLM |
| **D3 vs D7 model comparison** | Spec §14.7 — not implemented |
| **Seed list builder** with CSV download | Spec §14.8 — not a dedicated flow |

---

## 3. Code Quality Assessment

### Strengths

- **Clean separation of concerns**: Data layer (DuckDB), retrieval layer (TF-IDF), LLM layer (providers), rendering layer (ECharts) are clearly separated.
- **Robust error handling**: SQL validation, LLM error catching, model loading fallbacks, retry-on-SQL-error.
- **Streaming support**: All three LLM providers stream responses with a typing indicator (`▌`).
- **Report indexing**: TF-IDF index is built at startup (`@st.cache_resource`), chunked by `##` headings, with a relevance threshold (>0.05) to avoid irrelevant citations.
- **Security**: SQL blocklist is comprehensive. API keys are session-only. No file writes from user input.
- **Question history panel**: Fixed-position panel with scroll-to-question navigation — a nice UX touch not in the original spec.

### Weaknesses

- **Chart rendering is fragile**: ECharts specs are generated by the LLM as free-form JSON inside the response text. If the LLM formats it slightly differently (extra whitespace, nested objects, missing fields), parsing fails silently and no chart is shown.
- **No structured output**: The `CHARTS: [...]` pattern relies on regex extraction. More robust approaches: function calling / tool use, or a two-step LLM call (first for SQL, second for chart specs).
- **ECharts vs Plotly mismatch**: The rest of the app uses Plotly exclusively. The chatbot uses `streamlit-echarts`, introducing a visual inconsistency and an extra dependency.
- **No streaming of artifacts**: Charts and dataframes only render after the full response is streamed. For long responses this means a blank area below the text until streaming completes.
- **Report chunk size limit**: Chunks are capped at 3000 chars, snippets at 800 chars. Some reports have critical information beyond these limits.
- **Conversation history unbounded**: Last 20 messages are sent to LLM, but there's no token counting. Large SQL results in conversation history could exceed context limits.
- **pLTV inference flow**: The `run_pltv_inference` function uses categorical features as `pd.Categorical` type, but the training pipeline uses `LabelEncoder`. This mismatch could produce incorrect predictions.

---

## 4. Spec Compliance Score

| Category | Score | Notes |
|---|---|---|
| Core pipeline (classify → retrieve → SQL → render) | **9/10** | Solid implementation of the 6-step flow |
| LLM adapter | **10/10** | All 3 providers, streaming, clean API |
| DuckDB data layer | **9/10** | Safety, auto-limit, self-correction all work |
| Report retrieval | **8/10** | TF-IDF works but no BM25 option; chunk sizes could be tuned |
| Chart rendering | **6/10** | Multi-chart support exists but fragile; missing radar, no exports |
| pLTV inference | **7/10** | Works but feature encoding mismatch; no score distribution chart |
| Deterministic sub-flows | **3/10** | Only pLTV has a dedicated flow; 4 others are LLM-dependent |
| Export capabilities | **4/10** | CSV only; no HTML chart, no PNG, no canvas mode |
| Advanced analytics demos (§14) | **1/10** | None of the 8 showpiece features implemented |
| **Overall** | **~6.5/10** | Core pipeline is solid; advanced features and polish are the gaps |

---

## 5. Additional Features for On-the-Fly Analytics Demo

These features would transform the chatbot from a "SQL + text" tool into a **compelling analytics demo** that showcases what's possible with a well-defined user-level feature dataset like `cfm_pltv`.

### 5.1 Deterministic Analytics Modules (High Impact, Medium Effort)

These are pre-built, keyword-triggered analysis modules that run independently of LLM quality — guaranteeing correct, impressive outputs every time.

#### A. Live Pareto / Revenue Concentration Calculator
- **Trigger**: "pareto", "concentration", "top 1%", "revenue distribution"
- **Output**: Interactive Pareto chart with a slider for percentile threshold. User drags to see: "Top X% of users = Y% of revenue". Auto-annotates the whale threshold.
- **Why it demos well**: Stakeholders immediately see revenue concentration visually. The slider makes it interactive and memorable.

#### B. Late-Payer Revenue Opportunity Sizer
- **Trigger**: "late payer", "missed revenue", "D7-only", "opportunity"
- **Output**: Waterfall chart showing: D7 payer revenue → late payer revenue → total D30 revenue. Quantifies the $ gap. Compares D7-only seed vs enriched seed whale capture.
- **Why it demos well**: Directly answers "how much money are we leaving on the table?" — the #1 question UA managers ask.

#### C. Channel Deep-Dive Dashboard
- **Trigger**: "channel", "media source", "facebook", "google", "apple", "tiktok"
- **Output**: Multi-chart dashboard (no LLM needed): ARPU bar chart, payer rate bar chart, whale rate bar chart, late-payer rate bar chart, user volume bar chart — all grouped by `media_source`. Auto-highlights the best and worst channels.
- **Why it demos well**: Channel comparison is the most common ad-hoc analytics request in UA teams.

#### D. User Cohort Quality Tracker
- **Trigger**: "cohort quality", "user quality", "declining", "trend"
- **Output**: Line charts by `install_date` (weekly) showing ARPU, payer rate, whale rate trends. Auto-runs linear regression and reports the slope. Flags if quality is declining.
- **Why it demos well**: Shows the chatbot can detect trends and raise alerts, not just answer questions.

#### E. Feature Importance & Model Explainer
- **Trigger**: "feature importance", "what drives", "model", "xgboost", "explain"
- **Output**: Loads model metadata, renders feature importance bar chart, explains why `rev_d7` dominates (95.8%), and what the non-payment signals (gameplay, engagement) contribute for late-payer detection.
- **Why it demos well**: Non-technical stakeholders can understand the model without ML knowledge.

### 5.2 Interactive Analytics Widgets (High Impact, Higher Effort)

#### F. Segment Builder / Cohort Explorer
- **Trigger**: "segment", "filter", "cohort builder", "who are"
- **Output**: Multi-select filters (media_source, first_os, payer status, whale status, install date range) → instant KPI cards (count, ARPU, payer rate, whale rate, avg engagement) + downloadable user list.
- **Why it demos well**: Turns the chatbot into a self-service analytics tool. Non-technical users can slice data without SQL.

#### G. A/B Seed List Comparator
- **Trigger**: "seed", "compare seeds", "seed list", "UA seed"
- **Output**: Side-by-side comparison of 2-3 seed strategies with configurable parameters (top-k%, enrichment method). Shows whale capture, seed size, avg pLTV, and generates downloadable CSV seed lists.
- **Why it demos well**: Directly actionable — the output is a file that can be uploaded to an ad network.

#### H. User-Level pLTV Scorer with Explanation
- **Trigger**: "score user", "predict for", "individual", "vopenid"
- **Output**: For a specific user (by vopenid) or a filtered group, show: predicted pLTV, confidence interval (if available), top 3 features driving the prediction (SHAP-like), and comparison to population median.
- **Why it demos well**: Individual-level explanations make the model feel trustworthy and transparent.

### 5.3 Conversational Intelligence (Medium Impact, Low Effort)

#### I. Follow-Up Suggestion Engine
- After every response, auto-suggest 2-3 relevant follow-up questions based on the current result. Example: if user asked about payer rate by channel → suggest "Which channel has the highest whale rate?" and "Show late payer rate by channel".
- **Implementation**: Template-based, no LLM needed. Map query patterns to follow-up templates.

#### J. Insight Summarizer / Executive Brief Generator
- **Trigger**: "summary", "executive brief", "report", "key findings"
- **Output**: Aggregates key metrics (total users, revenue, payer rate, whale concentration, best/worst channel, model accuracy) into a formatted executive summary with KPI cards. Optionally exports as a formatted markdown/PDF.
- **Why it demos well**: Stakeholders get a one-page overview without asking multiple questions.

#### K. Anomaly / Outlier Highlighter
- **Trigger**: "anomaly", "outlier", "unusual", "suspicious"
- **Output**: Auto-detect outliers in revenue (extremely high LTV30 users), engagement (abnormally high game counts), or cohort quality (sudden drops). Flag and explain.
- **Why it demos well**: Shows the system can proactively surface interesting patterns.

### 5.4 Data Quality & Transparency (Low Impact, Low Effort but Trust-Building)

#### L. Data Freshness & Coverage Dashboard
- Always available via "data status", "data quality", "coverage"
- **Output**: Show dataset date range, row count, column completeness (% non-null per column), payer rate, revenue distribution stats. Helps users trust the data before asking questions.

#### M. SQL Explain Mode
- Toggle in sidebar: "Show SQL explanation"
- When enabled, the chatbot adds a plain-English explanation of what the SQL does, making it educational for non-technical users.

### 5.5 Export & Integration (Medium Impact, Medium Effort)

#### N. Chart Editor / Canvas Mode (from spec, not yet built)
- After each chart, show an "✏️ Edit" expander with the chart config JSON. User can modify titles, colors, chart type, and re-render. Demonstrates that the analytics output is not black-box.

#### O. Session Export
- "Export this conversation" → generates a formatted markdown or HTML file with all questions, answers, charts, and SQL used. Useful for sharing analysis with colleagues who weren't in the chat.

---

## 6. Priority Ranking for Demo Impact

If preparing for a demo with limited time, implement in this order:

| Priority | Feature | Effort | Demo Impact |
|---|---|---|---|
| **P0** | C. Channel Deep-Dive Dashboard | 2h | ★★★★★ |
| **P0** | A. Live Pareto Calculator | 2h | ★★★★★ |
| **P0** | B. Late-Payer Opportunity Sizer | 2h | ★★★★★ |
| **P1** | E. Feature Importance Explainer | 1h | ★★★★☆ |
| **P1** | J. Executive Brief Generator | 2h | ★★★★☆ |
| **P1** | I. Follow-Up Suggestions | 1h | ★★★★☆ |
| **P2** | D. Cohort Quality Tracker | 2h | ★★★☆☆ |
| **P2** | F. Segment Builder | 4h | ★★★★☆ |
| **P2** | G. Seed List Comparator | 3h | ★★★☆☆ |
| **P3** | H. User-Level Explainer | 4h | ★★★☆☆ |
| **P3** | L. Data Quality Dashboard | 1h | ★★☆☆☆ |
| **P3** | N. Chart Editor | 3h | ★★☆☆☆ |
| **P3** | O. Session Export | 2h | ★★☆☆☆ |

**Key insight**: The P0 features (Pareto, Late-Payer, Channel) are deterministic — they don't depend on LLM quality and will produce impressive, correct outputs every single time. This makes them the safest and most impactful additions for a live demo.

---

## 7. Technical Recommendations

1. **Fix pLTV feature encoding**: Use `LabelEncoder` (matching training) instead of `pd.Categorical` in `run_pltv_inference()`.
2. **Add structured output for charts**: Use LLM function calling or a two-step pipeline (SQL generation → chart spec generation) to improve chart reliability.
3. **Consider switching ECharts → Plotly**: Match the rest of the app, remove the `streamlit-echarts` dependency, and enable HTML/PNG export via `fig.to_html()` / `fig.to_image()`.
4. **Add token counting**: Before sending conversation history to LLM, estimate token count and truncate older messages to stay within context limits.
5. **Implement deterministic sub-flows first**: The 8 analytics demos from spec §14 don't require any LLM changes — they're pure Python + SQL + charting, triggered by keywords.
