# CFM Data Chatbot — Build Specification

## Context & Purpose

This page is the **showpiece demo** of the CrossFire Mobile (CFM) Vietnam pLTV analytics platform. It lets non-technical stakeholders (UA managers, product leads, monetisation teams) ask plain-English questions and get instant, data-grounded answers drawn from the actual feature table and the analysis reports we have already published. The chatbot must demonstrate our analytics depth: late-payer economics, whale concentration, channel quality, causal inference on engagement, and pLTV model scoring — all from a single conversational interface.

---

## 1. Where This Lives

- **File:** `webapp/pages/CFM_Chatbot.py`
- Follows the existing page naming convention used by `3b_Late_Payer_Analysis.py`, etc.
- Import `render_sidebar()` and `render_top_menu()` from `webapp/shared.py` exactly as every other page does.
- **Sidebar nav change:** In `shared.py → render_sidebar()`, rename the `🔧 Key Functions` section header to `🤖 CFM Data Chatbot` and replace / supplement `📤 Data Upload` + `📓 Notebooks` links with a prominent `page_link` to `CFM_Chatbot.py`. Keep the rest of the sidebar unchanged.
- **LLM adapter:** `webapp/utils/llm_providers.py` (create if missing).

---

## 2. Data Scope — Strict Grounding Rules

### Source of truth for CFM questions

| Source | Path | DuckDB alias |
|--------|------|--------------|
| Main feature table | `data/cfm_pltv_Feb22.csv` | `cfm_features` |
| Analysis reports | `reports/*.md` | Retrieved by keyword/TF-IDF |

**Dataset quick facts to embed in the system prompt:**
- 2,624,049 rows × 37 columns; installs Dec 16 2025 → Feb 21 2026; Vietnam only
- Revenue currency: VND (₫); 1 USD ≈ ₫24,000
- Target column: `ltv30` (30-day LTV in VND); binary payer flag: `is_payer_30`
- Late payer definition: `rev_d7 = 0 AND ltv30 > 0` (~2% of users, ~30% of total revenue)
- Whale definition: top 1% by LTV30 → 80.5% of all revenue (threshold ≈ ₫323,000)

### Available column groups (embed in system prompt)

```
Identity       : vopenid, roleid
Install        : install_date, game_id
Attribution    : media_source, campaign_id, adset_id, ad_id, site_id, first_os, last_os,
                 first_country_code, last_country_code, first_login_channel, last_login_channel
Login (D7)     : login_rows_d7, active_days_d7, loginchannel_variety_d7,
                 network_variety_d7, clientversion_variety_d7,
                 max_level_seen_d7, max_ladderscore_d7
Gameplay (D7)  : games_d7, win_rate_d7, avg_game_duration_d7, avg_score_d7,
                 kills_d7, deaths_d7, assists_d7, kd_d7,
                 max_level_game_d7, max_ladderlevel_d7
Revenue (D7)   : rev_d7, txn_cnt_d7, first_charge_day_offset_d7
Target/Label   : ltv30, is_payer_30
```

### Grounding rules

1. For any CFM question (metrics, revenue, payers, whales, channels, retention, pLTV, engagement, UA, cohorts) the assistant **must** answer using **only** `cfm_features` data and/or `reports/*.md` snippets.
2. The assistant **must not** hallucinate numbers. If a number is claimed, it must come from SQL execution result or a verbatim report citation.
3. If a question is about CFM but the required information cannot be derived from the CSV columns or reports, respond with:
   > "⚠️ Out of scope: I can't answer that from `cfm_pltv_Feb22.csv` or `reports/`. Here are the columns I can use: [list relevant column groups]."
4. For non-CFM questions, answer with general knowledge freely.

---

## 3. Data Execution Layer — DuckDB over CSV

```python
# Recommended initialisation (cached at startup)
@st.cache_resource
def get_duckdb_conn():
    import duckdb, pathlib
    con = duckdb.connect()
    csv_path = str(pathlib.Path(__file__).parents[2] / "data" / "cfm_pltv_Feb22.csv")
    con.execute(f"CREATE VIEW cfm_features AS SELECT * FROM read_csv_auto('{csv_path}')")
    return con
```

### Safety wrapper

- Strip and upper-case the SQL; reject if it starts with or contains `INSERT`, `UPDATE`, `DELETE`, `DROP`, `COPY`, `ATTACH`, `DETACH`, `PRAGMA`, `LOAD`, `IMPORT`, `EXPORT`.
- Auto-append `LIMIT 5000` if no LIMIT clause is present.
- Wrap execution in try/except; surface DuckDB errors as a friendly message with the SQL shown.

### Response anatomy (CFM mode)

1. **Short prose explanation** (2–4 sentences)
2. **Collapsible SQL block** (`st.expander("🔍 SQL used")`)
3. **DataFrame preview** (`st.dataframe`, max 200 rows displayed)
4. **Chart** (Plotly, auto-selected or user-requested) with export buttons
5. **Citations** (if any report was used)

---

## 4. Available Models — pLTV Inference Hook

Two production models exist. Load by scanning `models/*/metadata.json` and matching the directory name:

| Directory | Rows trained | Features | Notes |
|-----------|-------------|----------|-------|
| `pltv_model_20260223_14_16Mrows` | 1,655,607 | 14 | Primary; strongest on late-payer segment |
| `pltv_model_20260223_1543_11f_26Mrows` | 2,624,049 | 11 | Largest dataset; best generalisation |

Both models are **XGBoost Regressors** stored as `model.pkl` (pickle) with `metadata.json` alongside.

**Dominant feature:** `rev_d7` has ~95% feature importance in both models. The model's key value is identifying late payers (rev_d7=0) through gameplay/engagement signals.

### Inference flow

```python
# Pseudocode
model_dir = ROOT / "models" / selected_model_name
with open(model_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)
meta = json.load(open(model_dir / "metadata.json"))

# Build feature matrix from cfm_features query result
# Handle missing columns: fill numeric with 0, categorical with "unknown"
# Categorical features need pd.Categorical or label-encoding matching training
X = prepare_features(df, meta["numeric_features"], meta["categorical_features"])
df["pltv_score"] = model.predict(X)
```

- Return `pltv_score` column alongside original identifiers.
- Offer views: top-N by score, score distribution histogram, segmented by `media_source` / `first_os`.
- Clearly label results as **predictions, not actuals**.

---

## 5. Report Retrieval — BM25/TF-IDF over `reports/*.md`

### Reports inventory (23 files; key ones below)

| File | Core topic |
|------|-----------|
| `pLTV_Summary.md` | End-to-end model system; key metrics; business impact |
| `Temporal Analysis — Full Dataset…md` | 2.6M users Dec 16–Feb 21; ARPU decay; whale share; D7/D30 ratio |
| `Whale_Analysis_Overview.md` | Revenue concentration; 1%=80.5%; whale behavioural profile |
| `Causal_Inference.md` | Engagement → late conversion; dose-response; kills_d7 3.75× ratio |
| `Seed_Optimization_Strategy.md` | D7-only vs enriched seeds; whale capture; +10.9pp improvement |
| `Channel_Whale_Quality.md` | Per-channel whale rate; ASA vs Google gap |
| `Cohort_Comparison.md` | ARPU 2–3× across media sources |
| `Real_Time_Scoring.md` | D3 model retains ~97% of D7 accuracy |
| `Synthesis_Summary.md` | Cross-study executive summary |
| `Churn_Prediction.md` | Payer churn risk |
| `Whale_Segmentation.md` | Whale tier profiles |
| `Time_to_First_Purchase.md` | 78% of payers charge by D3 |

### Retrieval implementation

```python
# Simple TF-IDF at startup (no external deps)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def build_report_index():
    # Load all .md files; chunk by ## heading sections
    # Return (chunks_list, vectorizer, tfidf_matrix)

def retrieve_snippets(query: str, top_k: int = 3) -> list[dict]:
    # Returns [{"file": ..., "heading": ..., "snippet": ...}, ...]
```

### Citation format

> `[reports/pLTV_Summary.md § 3. Model Evaluation]`

Show a collapsible **Sources** panel at the bottom of each CFM response (`st.expander("📚 Sources used")`).

---

## 6. Chat UI — Streamlit

### Core layout

```python
render_top_menu()   # from shared.py — shows "🔥 CrossFire Decision Intelligence" header
render_sidebar()    # from shared.py — existing nav + dataset controls

# Main chat area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifact"):
            render_artifact(msg["artifact"])  # dataframe + chart + exports

if prompt := st.chat_input("Ask me about CFM payers, whales, revenue, pLTV…"):
    handle_message(prompt)
```

### Session state keys

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `list[dict]` | Full conversation history (role/content/artifact) |
| `chat_mode` | `"cfm"` \| `"general"` | Auto-detected; user-overridable toggle |
| `llm_provider` | `str` | Active provider |
| `llm_model` | `str` | Active model |
| `api_keys` | `dict` | `{provider: key}` — never logged |
| `duckdb_con` | resource | Cached DuckDB connection |
| `report_index` | resource | Cached TF-IDF report index |
| `pltv_model_cache` | `dict` | `{model_name: (model, meta)}` |

### Sidebar Settings expander

```
⚙️ LLM Settings
  Provider:  [ OpenAI | Anthropic | Gemini ]
  Model:     dropdown — per provider:
    OpenAI:    gpt-4o | gpt-4o-mini | gpt-3.5-turbo
    Anthropic: claude-3-5-sonnet-20241022 | claude-3-haiku-20240307 | claude-3-opus-20240229
    Gemini:    gemini-1.5-pro | gemini-1.5-flash | gemini-2.0-flash
  API Key:   password input (stored in session_state["api_keys"][provider])
```

Also add a sidebar `🗑️ Clear chat` button and a `📋 Show schema` button that opens an expander with all 37 column descriptions.

---

## 7. LLM Provider Adapter — `webapp/utils/llm_providers.py`

```python
def chat(
    provider: str,          # "openai" | "anthropic" | "gemini"
    model: str,
    api_key: str,
    messages: list[dict],   # [{"role": "user"|"assistant"|"system", "content": str}]
    stream: bool = False,
) -> str | Generator:
    """Unified LLM call. Returns full text or a streaming generator."""
```

**System prompt (injected automatically for CFM mode):**

```
You are the CFM Analytics Assistant for CrossFire Mobile Vietnam.

DATASET: cfm_pltv_Feb22.csv (alias: cfm_features) — 2,624,049 rows, 37 columns.
PERIOD: Dec 16, 2025 → Feb 21, 2026. Country: Vietnam (VN). Currency: VND (₫).

KEY FACTS YOU KNOW:
- Total revenue: ₫49.2B. Payer rate: 4.68%. Late payer rate: ~2%.
- Top 1% of users = 80.5% of revenue (whale threshold ≈ ₫323,000 LTV30).
- D7 revenue captures only ~29% of D30 revenue — D7-only ROAS is 3.4× understated.
- Late payers = 30% of total revenue; 34.6% of all whales.
- Best channel: Apple Search Ads (₫35,987 ARPU). Worst: Google Ads (₫6,257 ARPU).
- Model: XGBoost Regressor; rev_d7 has ~95.8% feature importance.
- D3 model retains ~97% of D7 Spearman ρ — viable for faster scoring.

COLUMNS: [full column list injected here]

RULES:
1. For CFM questions, generate DuckDB-compatible SQL over cfm_features, then answer from results.
2. Never hallucinate numbers. Cite the SQL result or the report section.
3. If unanswerable from the dataset/reports, refuse clearly and suggest alternatives.
4. For general analytics/ML questions, answer freely with general knowledge.
```

---

## 8. Analytics-Native Responses: Charts + Export

### Chart selection logic

| Query pattern | Suggested chart |
|---|---|
| Distribution / histogram | Plotly histogram or box plot |
| Trend over time (`install_date`) | Line chart |
| Comparison across groups (`media_source`, `first_os`) | Bar chart |
| Correlation / scatter | Scatter plot |
| Funnel / stage (payer rate, whale rate) | Funnel or stacked bar |
| Concentration / Pareto | Sorted bar + cumulative line |
| pLTV score distribution | Histogram + percentile lines |

### Export buttons (below every chart/table artifact)

```python
# CSV export
st.download_button("⬇️ Export CSV", df.to_csv(index=False), "cfm_result.csv", "text/csv")

# HTML export (standalone Plotly)
html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
st.download_button("⬇️ Export Chart HTML", html_str, "cfm_chart.html", "text/html")

# Image export (graceful skip if kaleido not installed)
try:
    img_bytes = fig.to_image(format="png")
    st.download_button("⬇️ Export PNG", img_bytes, "cfm_chart.png", "image/png")
except Exception:
    pass
```

### Canvas mode for charts

For responses that produce a chart, add an **"✏️ Edit chart"** expander beneath it. Inside, render a `st.text_area` pre-filled with the Plotly figure JSON or Python snippet so the user can modify axis labels, color schemes, chart type, etc., and press "Re-render" to apply. This demonstrates the "analytics canvas" concept without requiring a full code execution environment.

---

## 9. Conversation Flow — Deterministic CFM Controller

```
User question
     │
     ▼
Step A: CLASSIFY
  keywords → CFM: ltv, pltv, payer, whale, arpu, roas, revenue, cohort, install,
             channel, media_source, games, active_days, kills, churn, seed, model,
             score, predict, cfm, crossfire, d7, d30, late payer, engagement
  → if match → CFM mode
  → else     → General mode
     │
     ▼ (CFM mode)
Step B: RETRIEVE REPORTS
  TF-IDF search → top 3 snippets from reports/*.md
  Attach to LLM context as grounding material
     │
     ▼
Step C: GENERATE SQL
  LLM receives: system prompt + schema + report snippets + user question
  LLM outputs: SQL query targeting cfm_features
  Validate SQL (safety check)
     │
     ▼
Step D: EXECUTE SQL
  DuckDB → DataFrame result
  If error → retry once with error feedback to LLM (self-correction)
     │
     ▼
Step E: GENERATE RESPONSE
  LLM receives: SQL result (as markdown table, max 20 rows shown) + snippets
  LLM outputs: prose explanation
     │
     ▼
Step F: RENDER ARTIFACT
  st.dataframe (full result) + chart + export buttons + citations
```

### Special sub-flows

**pLTV scoring flow** (triggered by: "score", "predict", "pLTV", "model", "top 1%")
1. Let user pick model from sidebar or inline selector.
2. Query `cfm_features` for required feature columns (handle nulls → 0 / "unknown").
3. Run `model.predict(X)` → append `pltv_score` column.
4. Show top-N leaderboard + score distribution chart.
5. Offer export of the full scored table.

**Whale analysis flow** (triggered by: "whale", "top 1%", "concentration", "pareto")
- Auto-compute whale threshold at P99 of `ltv30`.
- Show Pareto chart: cumulative % users vs cumulative % revenue.
- Break down by `media_source` and `first_os`.

**Channel comparison flow** (triggered by: "channel", "media_source", "facebook", "google", "tiktok", "apple")
- Group by `media_source`; compute ARPU, payer_rate, late_payer_rate, whale_rate.
- Ranked bar chart coloured by ARPU.

**Late payer flow** (triggered by: "late payer", "d7 non-payer", "after d7")
- Filter `rev_d7 = 0 AND ltv30 > 0`; compare behavioural features vs non-payers.
- Cite `Causal_Inference.md` and `pLTV_Summary.md §6.1` automatically.

---

## 10. Suggested Starter Questions (Sidebar)

Display these as clickable pills in the sidebar (`st.button`) to populate the chat input:

```
📊  "What is the payer rate and ARPU by media source?"
🐋  "Show me the whale concentration — top 1% vs rest"
⏱️  "How does D7 revenue compare to D30 revenue across cohorts?"
🔍  "Who are the late payers and what behavioral signals predict them?"
🎯  "Score all users with the pLTV model and show the top 1% predicted"
📉  "Which channels have declining cohort quality over time?"
🌱  "Compare seed strategies: D7-payers-only vs enriched with late payers"
💡  "What does the causal inference analysis say about engagement driving conversion?"
```

---

## 11. Practical Implementation Notes

### Performance

- Use `@st.cache_resource` for DuckDB connection and report index (process-level, shared across reruns).
- Use `@st.cache_data` for any heavy pandas operations keyed by (query_hash, limit).
- Never load the full 527 MB CSV into a pandas DataFrame eagerly — DuckDB reads it lazily via `read_csv_auto`.
- Default query limit: 5,000 rows. For ML inference, cap at 100,000 rows via a sidebar slider.

### Security

- API keys stored in `st.session_state["api_keys"]` only; never written to disk, never logged.
- SQL safety check: reject destructive statements before passing to DuckDB.
- No external web calls for CFM answers.

### Error handling

- DuckDB SQL error → show error + SQL + "Try rephrasing your question".
- Missing model file → graceful warning with list of available models.
- LLM API error → show provider error message + suggest checking API key.
- Missing columns for model → fill with 0/unknown and warn user which columns were imputed.

### Currency toggle

- Respect `st.session_state["currency"]` (VND / USD) set by the existing sidebar currency selector.
- Use `format_currency()` from `shared.py` for all displayed monetary values.

---

## 12. File Structure Summary

```
webapp/
  pages/
    CFM_Chatbot.py          ← NEW: main chatbot page
  utils/
    llm_providers.py        ← NEW: multi-provider LLM adapter
  shared.py                 ← MODIFIED: add chatbot link to sidebar nav
```

No new dependencies required beyond what is already in `webapp/requirements.txt` (openai, anthropic, google-generativeai must be added; duckdb must be added).

**New requirements to add to `requirements.txt`:**
```
duckdb>=0.10.0
openai>=1.30.0
anthropic>=0.28.0
google-generativeai>=0.7.0
```

---

## 13. Acceptance Tests

| Prompt | Expected behaviour |
|--------|--------------------|
| `"Compute payer rate top 10% and plot distribution"` | SQL executes, shows table + Plotly histogram, CSV + HTML export work |
| `"What is the D7 feature window in this dataset?"` | Answers from schema description + cites `pLTV_Summary.md §2` |
| `"How many cheaters last week in CFM?"` | Refuses: column not in schema; suggests usable columns |
| `"General question: what is ARPPU?"` | General mode: answers from general knowledge, no SQL |
| `"Score pLTV for users and show top 1% predicted"` | Loads pickle model, runs inference, shows top-1% leaderboard + score distribution |
| `"Compare whale rate across media sources"` | SQL groups by media_source, whale rate bar chart, cites Channel_Whale_Quality.md |
| `"Show late payer rate trend by install week"` | SQL groups install_date by week, line chart, cites Causal_Inference.md |
| `"What does our analysis say about D3 scoring?"` | Retrieves Real_Time_Scoring.md, answers from report, no SQL needed |
| `"Which channel has the best ARPU?"` | SQL → Apple Search Ads ₫35,987; cites temporal analysis report |
| `"Explain the seed optimization strategy"` | Retrieves Seed_Optimization_Strategy.md; explains D7-only vs enriched; shows whale capture table |

---

## 14. Additional Intelligence Demos to Highlight

These are analytics showpieces that make the chatbot feel genuinely powerful:

1. **Pareto live calculator** — user asks "show me the Pareto" → generates real cumulative revenue chart with dynamic percentile threshold slider.

2. **Late-payer opportunity quantifier** — user asks "how much revenue am I leaving on the table with D7-only seeds?" → SQL computes missed revenue vs oracle seed, formatted as a business case.

3. **Cohort quality trend alert** — user asks "is our user quality declining?" → runs regression on ARPU by `install_date`, reports slope (−₫326/day confirmed in dataset), plots trend line.

4. **Feature importance explainer** — user asks "what drives pLTV?" → loads model metadata, renders bar chart of feature importances from `metadata.json`, explains why `rev_d7` dominates (95.8%) and what the non-payment signals contribute.

5. **Channel ROI ranker** — user asks "which channel gives the best ROAS?" → computes ARPU / (assumed CPI) proxy ranking, formats as a sortable table with recommended action per channel.

6. **Engagement → conversion funnel** — user asks "how does game engagement relate to late conversion?" → SQL buckets `games_d7` into quintiles, computes late_payer_rate per bucket, shows dose-response chart (mirrors Causal_Inference.md finding: 8.86× lift for high-engagement users).

7. **D3 vs D7 model comparison** — user asks "can we score users earlier?" → explains D3 viability from Real_Time_Scoring.md, applies model to subset with simulated D3 features (games/active_days scaled by 3/7), shows correlation.

8. **Seed list builder** — user asks "give me a UA seed list" → produces CSV download of top-N users by pLTV score, segmented by media_source, ready to upload to an ad network.
