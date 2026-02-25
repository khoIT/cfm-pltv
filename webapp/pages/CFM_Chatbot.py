"""
CFM Data Chatbot — Conversational analytics over the CFM pLTV feature table.
Lets users ask plain-English questions and get data-grounded answers from
cfm_pltv_Feb22.csv and reports/*.md, with SQL execution, charts, and exports.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import re
import textwrap
from pathlib import Path
from typing import Optional
from streamlit_echarts import st_echarts
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import (
    render_sidebar,
    render_top_menu,
    ROOT,
    REPORTS_DIR,
    format_currency,
    get_currency_info,
    convert_vnd,
    FEATURE_GROUPS,
)
from utils.analytics_modules import (
    detect_deterministic_flow,
    run_pareto_analysis,
    run_late_payer_analysis,
    run_channel_dashboard,
    run_cohort_tracker,
    run_feature_importance,
    run_executive_brief,
    run_anomaly_detection,
    run_data_quality,
    get_follow_up_suggestions,
    render_follow_ups,
)

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
render_top_menu()
render_sidebar()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Auto-detect the main data CSV (the old hardcoded name may not exist)
def _find_csv():
    candidates = [
        DATA_DIR / "cfm_pltv_Feb22.csv",
        DATA_DIR / "cfm_pltv_train.csv",
        DATA_DIR / "cfm_pltv_2025_12_16.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: first cfm_pltv*.csv found
    globbed = sorted(DATA_DIR.glob("cfm_pltv*.csv"))
    return globbed[0] if globbed else candidates[0]

CSV_PATH = _find_csv()

CFM_KEYWORDS = {
    "ltv", "pltv", "payer", "whale", "arpu", "arppu", "roas", "revenue",
    "cohort", "install", "channel", "media_source", "games", "active_days",
    "kills", "churn", "seed", "model", "score", "predict", "cfm", "crossfire",
    "d7", "d30", "late payer", "engagement", "retention", "monetisation",
    "monetization", "payment", "conversion", "campaign", "ua", "user acquisition",
    "login", "kd", "win_rate", "deaths", "assists", "first_os", "country",
    "install_date", "pareto", "concentration", "non-payer", "non payer",
    "feature importance", "xgboost",
}

SQL_BLOCKLIST = {"INSERT", "UPDATE", "DELETE", "DROP", "COPY", "ATTACH",
                 "DETACH", "PRAGMA", "LOAD", "IMPORT", "EXPORT", "ALTER",
                 "CREATE", "TRUNCATE", "GRANT", "REVOKE"}

COLUMN_SCHEMA = """
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
""".strip()

SYSTEM_PROMPT = textwrap.dedent(f"""\
You are the CFM Analytics Assistant for CrossFire Mobile Vietnam.
Your PRIMARY job is to deliver clear, actionable INSIGHTS and DATA STORIES to the user.

DATASET: {CSV_PATH.name} (DuckDB table alias: cfm_features) — 37 columns.
PERIOD: Dec 16, 2025 → Feb 21, 2026. Country: Vietnam (VN). Currency: VND (₫). 1 USD ≈ ₫24,000.

KEY FACTS:
- Total revenue: ₫49.2B. Payer rate: 4.68%. Late payer rate: ~2%.
- Top 1% of users = 80.5% of revenue (whale threshold ≈ ₫323,000 LTV30).
- D7 revenue captures only ~29% of D30 revenue — D7-only ROAS is 3.4× understated.
- Late payers (rev_d7=0, ltv30>0) = 30% of total revenue; 34.6% of all whales.
- Best channel: Apple Search Ads (₫35,987 ARPU). Worst: Google Ads (₫6,257 ARPU).
- Model: XGBoost Regressor; rev_d7 has ~95.8% feature importance.
- D3 model retains ~97% of D7 Spearman ρ — viable for faster scoring.

COLUMNS:
{COLUMN_SCHEMA}

RESPONSE FORMAT — follow this structure for every CFM answer:

1. **Insight first**: Start with a clear, business-oriented answer to the user's question (3-6 sentences). Explain what the data means, why it matters, and what action the user could take. This is the most important part.

2. **Multiple charts** (REQUIRED for data questions): You MUST generate 3-4 charts to illustrate different angles of the answer. Output a JSON array on a line starting with:
   CHARTS: [{{"type":"bar","x":"col","y":"col","title":"..."}}, {{"type":"line","x":"col","y":"col","title":"..."}}, ...]
   Supported chart types: bar, line, scatter, pie, heatmap, radar, funnel, boxplot.
   Each object must have: type, title. Plus x and y (column names from your SQL output).
   For pie charts use "names" and "values" instead of x/y.
   ALWAYS generate at least 1 chart, ideally 3-4 to show different perspectives (e.g. a bar chart for comparison, a pie for composition, a line for trends).

3. **SQL query** (secondary, for reference): When you need data from the table, output the SQL inside a ```sql code fence. The user can optionally review it — it is NOT the primary deliverable. Keep SQL clean and well-aliased so the result columns are human-readable. Include enough columns to support all your charts.

RULES:
1. ALWAYS lead with insight and interpretation, not raw data or SQL. The user wants to understand their business, not read queries.
2. Never hallucinate numbers. Only state numbers from the SQL result or report citations provided.
3. If the question cannot be answered from cfm_features columns or the reports, refuse clearly: "⚠️ Out of scope" and suggest what columns ARE available.
4. For general analytics/ML knowledge questions unrelated to CFM data, answer freely.
5. When citing report findings, name the report file and section.
6. Proactively suggest follow-up questions or deeper dives when relevant.
7. ALWAYS include charts — visualizations are the primary artifact alongside your insight text.
""")


# ---------------------------------------------------------------------------
# DuckDB connection (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_duckdb_conn():
    import duckdb
    con = duckdb.connect()
    csv = str(CSV_PATH).replace("\\", "/")
    con.execute(f"CREATE VIEW cfm_features AS SELECT * FROM read_csv_auto('{csv}')")
    return con


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (is_safe, message). Rejects destructive SQL."""
    tokens = set(re.findall(r'\b[A-Z]+\b', sql.upper()))
    blocked = tokens & SQL_BLOCKLIST
    if blocked:
        return False, f"Blocked SQL keywords: {', '.join(blocked)}"
    return True, ""


def execute_sql(sql: str, limit: int = 5000) -> tuple[Optional[pd.DataFrame], str]:
    """Execute SQL via DuckDB. Returns (df | None, error_message)."""
    safe, msg = validate_sql(sql)
    if not safe:
        return None, msg
    # Auto-append LIMIT if missing
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + f"\nLIMIT {limit}"
    try:
        con = get_duckdb_conn()
        df = con.execute(sql).fetchdf()
        return df, ""
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Report retrieval (TF-IDF, cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def build_report_index():
    """Load all .md files from reports/ AND root-level analysis docs, chunk by ## headings, build TF-IDF index."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Collect .md files from multiple locations
    SKIP_FILES = {"README.md", "README_DEPLOYMENT.md", "chatbot-build-prompt.md",
                  "CURRENCY_TOGGLE_IMPLEMENTATION.md", "NOTEBOOK_UPDATE_GUIDE.md"}
    md_paths = []
    # reports/ directory
    md_paths.extend(sorted(REPORTS_DIR.glob("*.md")))
    # Root-level analysis docs
    for md_path in sorted(ROOT.glob("*.md")):
        if md_path.name not in SKIP_FILES:
            md_paths.append(md_path)

    chunks = []  # list of {"file": str, "heading": str, "text": str}
    for md_path in md_paths:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r'^(##\s+.+)$', text, flags=re.MULTILINE)
        # sections = [preamble, heading1, body1, heading2, body2, ...]
        current_heading = "Introduction"
        buf = ""
        for part in sections:
            if part.startswith("## "):
                if buf.strip():
                    chunks.append({
                        "file": md_path.name,
                        "heading": current_heading,
                        "text": buf.strip()[:3000],
                    })
                current_heading = part.strip("# ").strip()
                buf = ""
            else:
                buf += part + "\n"
        if buf.strip():
            chunks.append({
                "file": md_path.name,
                "heading": current_heading,
                "text": buf.strip()[:3000],
            })

    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return chunks, vectorizer, tfidf_matrix


def retrieve_snippets(query: str, top_k: int = 3) -> list[dict]:
    """Return top-k report snippets relevant to the query."""
    from sklearn.metrics.pairwise import cosine_similarity

    chunks, vectorizer, tfidf_matrix = build_report_index()
    if not chunks:
        return []
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        if scores[i] > 0.05:
            results.append({
                "file": chunks[i]["file"],
                "heading": chunks[i]["heading"],
                "snippet": chunks[i]["text"][:800],
                "score": float(scores[i]),
            })
    return results


# ---------------------------------------------------------------------------
# pLTV Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pltv_model(model_name: str):
    """Load a pickled XGBoost model + metadata from models/<model_name>/."""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        return None, None
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    meta = json.loads((model_dir / "metadata.json").read_text())
    return model, meta


def list_available_models() -> list[str]:
    """Return directory names under models/ that have model.pkl + metadata.json."""
    models = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir() and (p / "model.pkl").exists() and (p / "metadata.json").exists():
                models.append(p.name)
    return models


def run_pltv_inference(model_name: str, max_rows: int = 50000) -> tuple[Optional[pd.DataFrame], str]:
    """Score users from cfm_features using selected pLTV model."""
    model, meta = load_pltv_model(model_name)
    if model is None:
        return None, f"Model {model_name} not found."

    num_feats = meta.get("numeric_features", [])
    cat_feats = meta.get("categorical_features", [])
    all_feats = num_feats + cat_feats

    # Query only needed columns
    cols_sql = ", ".join(["vopenid", "media_source", "first_os", "ltv30", "rev_d7"] +
                         [c for c in all_feats if c not in ("vopenid", "media_source", "first_os", "ltv30", "rev_d7")])
    sql = f"SELECT {cols_sql} FROM cfm_features LIMIT {max_rows}"
    df, err = execute_sql(sql, limit=max_rows)
    if err:
        return None, err

    # Prepare feature matrix
    X = pd.DataFrame()
    for c in num_feats:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            X[c] = 0
    for c in cat_feats:
        if c in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[c] = le.fit_transform(df[c].fillna("unknown").astype(str))
        else:
            X[c] = 0

    try:
        df["pltv_score"] = model.predict(X)
    except Exception as e:
        return None, f"Prediction error: {e}"

    return df, ""


# ---------------------------------------------------------------------------
# Classification: CFM mode vs General mode
# ---------------------------------------------------------------------------
def classify_intent(question: str) -> str:
    """Return 'cfm' or 'general' based on keyword matching."""
    q_lower = question.lower()
    for kw in CFM_KEYWORDS:
        if kw in q_lower:
            return "cfm"
    return "general"


# ---------------------------------------------------------------------------
# Parse LLM response for SQL and chart specs
# ---------------------------------------------------------------------------
def parse_llm_response(text: str) -> dict:
    """
    Extract SQL block, chart specs (array), and prose from the LLM output.
    Returns {"prose": str, "sql": str|None, "charts": list[dict]}
    """
    result = {"prose": text, "sql": None, "charts": []}

    # Extract SQL
    sql_match = re.search(r'```sql\s*\n(.*?)```', text, re.DOTALL)
    if sql_match:
        result["sql"] = sql_match.group(1).strip()
        result["prose"] = text[:sql_match.start()] + text[sql_match.end():]

    # Extract CHARTS: [...] (new multi-chart format)
    charts_match = re.search(r'CHARTS:\s*(\[.*?\])', result["prose"], re.DOTALL)
    if charts_match:
        try:
            specs = json.loads(charts_match.group(1))
            if isinstance(specs, list):
                result["charts"] = specs
            result["prose"] = result["prose"][:charts_match.start()] + result["prose"][charts_match.end():]
        except json.JSONDecodeError:
            pass

    # Fallback: also try old single CHART_SPEC: {...} format
    if not result["charts"]:
        chart_match = re.search(r'CHART_SPEC:\s*(\{.*?\})', result["prose"], re.DOTALL)
        if chart_match:
            try:
                spec = json.loads(chart_match.group(1))
                result["charts"] = [spec]
                result["prose"] = result["prose"][:chart_match.start()] + result["prose"][chart_match.end():]
            except json.JSONDecodeError:
                pass

    result["prose"] = result["prose"].strip()
    return result


# ---------------------------------------------------------------------------
# ECharts rendering from spec + DataFrame
# ---------------------------------------------------------------------------
def build_echarts_option(df: pd.DataFrame, spec: dict) -> Optional[dict]:
    """Build an ECharts option dict from a DataFrame and a chart spec."""
    if not spec or df is None or df.empty:
        return None

    chart_type = spec.get("type", "bar").lower()
    title = spec.get("title", "")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    names = spec.get("names")  # for pie
    values = spec.get("values")  # for pie

    try:
        if chart_type == "pie":
            n_col = names or x
            v_col = values or y
            if not n_col or not v_col or n_col not in df.columns or v_col not in df.columns:
                return None
            pie_data = [{"name": str(row[n_col]), "value": float(row[v_col])}
                        for _, row in df.iterrows() if pd.notna(row[v_col])]
            return {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "legend": {"orient": "vertical", "left": "left", "top": "middle"},
                "series": [{"type": "pie", "radius": ["40%", "70%"], "data": pie_data,
                            "emphasis": {"itemStyle": {"shadowBlur": 10}}}],
            }

        if not x or not y or x not in df.columns or y not in df.columns:
            return None

        x_data = df[x].astype(str).tolist()
        y_data = [float(v) if pd.notna(v) else 0 for v in df[y]]

        base_option = {
            "title": {"text": title, "left": "center"},
            "tooltip": {"trigger": "axis" if chart_type != "scatter" else "item"},
            "grid": {"left": "10%", "right": "5%", "bottom": "15%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}},
        }

        if chart_type in ("bar", "line"):
            base_option["xAxis"] = {"type": "category", "data": x_data,
                                     "axisLabel": {"rotate": 30 if len(x_data) > 8 else 0}}
            base_option["yAxis"] = {"type": "value"}
            series_item = {"type": chart_type, "data": y_data, "name": y}
            if chart_type == "bar":
                series_item["itemStyle"] = {"borderRadius": [4, 4, 0, 0]}
                series_item["colorBy"] = "data"
            if chart_type == "line":
                series_item["smooth"] = True
                series_item["areaStyle"] = {"opacity": 0.15}
            # If color column exists, build grouped series
            if color and color in df.columns:
                groups = df.groupby(color)
                series = []
                for name, grp in groups:
                    series.append({
                        "type": chart_type, "name": str(name),
                        "data": [float(v) if pd.notna(v) else 0 for v in grp[y]],
                    })
                base_option["xAxis"]["data"] = grp[x].astype(str).tolist()
                base_option["legend"] = {"data": [str(n) for n in df[color].unique()]}
                base_option["series"] = series
            else:
                base_option["series"] = [series_item]
            return base_option

        if chart_type == "scatter":
            scatter_data = [[float(row[x]) if pd.notna(row[x]) else 0,
                             float(row[y]) if pd.notna(row[y]) else 0]
                            for _, row in df.iterrows()]
            base_option["xAxis"] = {"type": "value", "name": x}
            base_option["yAxis"] = {"type": "value", "name": y}
            base_option["series"] = [{"type": "scatter", "data": scatter_data,
                                       "symbolSize": 8}]
            return base_option

        if chart_type == "funnel":
            funnel_data = [{"name": str(row[x]), "value": float(row[y])}
                           for _, row in df.iterrows() if pd.notna(row[y])]
            base_option["series"] = [{"type": "funnel", "data": funnel_data,
                                       "sort": "descending", "gap": 2}]
            return base_option

        if chart_type == "heatmap":
            base_option["xAxis"] = {"type": "category", "data": x_data}
            base_option["yAxis"] = {"type": "category"}
            base_option["visualMap"] = {"min": min(y_data), "max": max(y_data),
                                         "calculable": True}
            heat_data = [[i, 0, v] for i, v in enumerate(y_data)]
            base_option["series"] = [{"type": "heatmap", "data": heat_data}]
            return base_option

        if chart_type == "radar":
            # Radar chart: x = category labels, y = values
            indicators = [{"name": str(v), "max": float(df[y].max()) * 1.2}
                          for v in df[x].tolist()]
            radar_data = [float(v) if pd.notna(v) else 0 for v in df[y]]
            base_option.pop("grid", None)
            base_option["radar"] = {"indicator": indicators, "shape": "polygon"}
            base_option["series"] = [{
                "type": "radar",
                "data": [{"value": radar_data, "name": y}],
                "areaStyle": {"opacity": 0.2},
            }]
            return base_option

        if chart_type == "boxplot":
            # Simple boxplot from y values grouped by x
            base_option["xAxis"] = {"type": "category", "data": list(df[x].unique().astype(str))}
            base_option["yAxis"] = {"type": "value"}
            box_data = []
            for cat in df[x].unique():
                vals = df[df[x] == cat][y].dropna().tolist()
                if vals:
                    vals.sort()
                    n = len(vals)
                    box_data.append([min(vals), vals[n // 4], vals[n // 2],
                                      vals[3 * n // 4], max(vals)])
            base_option["series"] = [{"type": "boxplot", "data": box_data}]
            return base_option

        # Fallback: bar chart
        base_option["xAxis"] = {"type": "category", "data": x_data}
        base_option["yAxis"] = {"type": "value"}
        base_option["series"] = [{"type": "bar", "data": y_data, "name": y}]
        return base_option

    except Exception:
        return None


def render_echarts(df: pd.DataFrame, charts: list[dict], key_suffix: str = ""):
    """Render multiple ECharts from a list of chart specs + DataFrame."""
    rendered = 0
    for idx, spec in enumerate(charts):
        option = build_echarts_option(df, spec)
        if option:
            st_echarts(option, height="420px", key=f"echart_{key_suffix}_{idx}")
            rendered += 1
    return rendered


def auto_generate_charts(df: pd.DataFrame) -> list[dict]:
    """Auto-generate chart specs when the LLM didn't provide any."""
    charts = []
    if df is None or df.empty or len(df.columns) < 2:
        return charts

    str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if str_cols and num_cols and len(df) <= 50 and len(df) > 1:
        # Bar chart: first string col vs first numeric col
        charts.append({"type": "bar", "x": str_cols[0], "y": num_cols[0],
                        "title": f"{num_cols[0]} by {str_cols[0]}"})
        # If there's a second numeric col, add a line chart
        if len(num_cols) >= 2:
            charts.append({"type": "line", "x": str_cols[0], "y": num_cols[1],
                            "title": f"{num_cols[1]} trend by {str_cols[0]}"})
        # Pie chart if few categories
        if len(df) <= 15:
            charts.append({"type": "pie", "names": str_cols[0], "values": num_cols[0],
                            "title": f"{num_cols[0]} composition"})

    elif len(num_cols) >= 2 and not str_cols:
        # Scatter if all numeric
        charts.append({"type": "scatter", "x": num_cols[0], "y": num_cols[1],
                        "title": f"{num_cols[1]} vs {num_cols[0]}"})
    return charts


# ---------------------------------------------------------------------------
# Render an artifact (dataframe + ECharts + exports)
# ---------------------------------------------------------------------------
def render_artifact(df: Optional[pd.DataFrame], charts: list[dict], key_suffix: str = ""):
    """Render dataframe, multiple ECharts, export buttons, and chart editor."""
    if df is not None and not df.empty:
        st.dataframe(df.head(200), use_container_width=True)

        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Export CSV", csv_data, f"cfm_result_{key_suffix}.csv", "text/csv",
            key=f"csv_dl_{key_suffix}",
        )

        # Render all charts
        if charts:
            render_echarts(df, charts, key_suffix=key_suffix)

            # Chart editor / canvas mode
            with st.expander("✏️ Edit Charts (Canvas Mode)", expanded=False):
                chart_json = json.dumps(charts, indent=2)
                edited = st.text_area(
                    "Modify chart specs (JSON array) and click Re-render:",
                    value=chart_json, height=180,
                    key=f"canvas_{key_suffix}",
                )
                if st.button("🔄 Re-render", key=f"rerender_{key_suffix}"):
                    try:
                        new_specs = json.loads(edited)
                        if isinstance(new_specs, list):
                            render_echarts(df, new_specs, key_suffix=f"{key_suffix}_edited")
                            st.success("Charts re-rendered with your edits.")
                        else:
                            st.error("Must be a JSON array of chart specs.")
                    except json.JSONDecodeError as je:
                        st.error(f"Invalid JSON: {je}")


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------
def call_llm(messages: list[dict], stream: bool = True):
    """Call the configured LLM. Returns full text or generator."""
    from utils.llm_providers import chat

    provider = st.session_state.get("llm_provider", "openai")
    model = st.session_state.get("llm_model", "gpt-4o-mini")
    api_keys = st.session_state.get("api_keys", {})
    api_key = api_keys.get(provider, "")

    if not api_key:
        raise ValueError(f"No API key set for {provider}. Please enter it in the sidebar settings.")

    return chat(provider=provider, model=model, api_key=api_key, messages=messages, stream=stream)


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------
def _run_deterministic_flow(flow: str, user_input: str) -> tuple[str, list[str]]:
    """Dispatch to the appropriate deterministic analytics module.
    Returns (summary_text, follow_up_suggestions)."""
    dispatch = {
        "pareto": lambda: run_pareto_analysis(execute_sql),
        "late_payer": lambda: run_late_payer_analysis(execute_sql),
        "channel": lambda: run_channel_dashboard(execute_sql),
        "cohort": lambda: run_cohort_tracker(execute_sql),
        "feature_importance": lambda: run_feature_importance(execute_sql, MODELS_DIR),
        "executive_brief": lambda: run_executive_brief(execute_sql),
        "anomaly": lambda: run_anomaly_detection(execute_sql),
        "data_quality": lambda: run_data_quality(execute_sql),
    }
    handler = dispatch.get(flow)
    if handler:
        return handler()
    return "Unknown flow.", []


def handle_message(user_input: str):
    """Process a user message through the full chatbot pipeline.
    Priority: deterministic flow → LLM-based response."""
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    mode = classify_intent(user_input)
    st.session_state.chat_mode = mode

    # ── Step 1: Check for deterministic analytics flow ──────────
    flow = detect_deterministic_flow(user_input)
    if flow and mode == "cfm":
        try:
            with st.chat_message("assistant"):
                summary, follow_ups = _run_deterministic_flow(flow, user_input)

            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_follow_ups = follow_ups
            return
        except Exception as e:
            # If deterministic flow fails, fall through to LLM
            st.warning(f"Analytics module error: {e} — falling back to LLM.")

    # ── Step 2: LLM-based response ─────────────────────────────
    # Check for pLTV scoring intent
    pltv_keywords = {"score", "predict", "pltv", "inference", "model score"}
    is_pltv_request = any(kw in user_input.lower() for kw in pltv_keywords)

    # Build LLM messages
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # If CFM mode, retrieve relevant report snippets
    snippets = []
    if mode == "cfm":
        snippets = retrieve_snippets(user_input, top_k=3)
        if snippets:
            report_context = "\n\nRELEVANT REPORT EXCERPTS:\n"
            for s in snippets:
                report_context += f"\n--- [{s['file']} § {s['heading']}] ---\n{s['snippet']}\n"
            llm_messages[0]["content"] += report_context

    # Add conversation history (last 10 exchanges for context)
    history = st.session_state.messages[:-1]  # exclude the just-added user msg
    for msg in history[-20:]:
        role = msg["role"]
        content = msg["content"]
        if role in ("user", "assistant"):
            llm_messages.append({"role": role, "content": content})

    llm_messages.append({"role": "user", "content": user_input})

    # Call LLM
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            stream = call_llm(llm_messages, stream=True)
            for chunk in stream:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Parse the response
            parsed = parse_llm_response(full_response)

            df_result = None
            sql_used = parsed.get("sql")
            chart_specs = parsed.get("charts", [])

            # Execute SQL if present
            if sql_used:
                with st.expander("🔍 SQL used", expanded=False):
                    st.code(sql_used, language="sql")

                df_result, err = execute_sql(sql_used)
                if err:
                    # Retry: send error back to LLM for self-correction
                    retry_msg = llm_messages + [
                        {"role": "assistant", "content": full_response},
                        {"role": "user", "content": f"The SQL failed with error: {err}\nPlease fix the SQL and try again."},
                    ]
                    try:
                        retry_response = ""
                        retry_stream = call_llm(retry_msg, stream=True)
                        retry_placeholder = st.empty()
                        for chunk in retry_stream:
                            retry_response += chunk
                            retry_placeholder.markdown(retry_response + "▌")
                        retry_placeholder.markdown(retry_response)

                        retry_parsed = parse_llm_response(retry_response)
                        if retry_parsed.get("sql"):
                            sql_used = retry_parsed["sql"]
                            with st.expander("🔍 Corrected SQL", expanded=False):
                                st.code(sql_used, language="sql")
                            df_result, err2 = execute_sql(sql_used)
                            if err2:
                                st.error(f"SQL error: {err2}")
                            if retry_parsed.get("charts"):
                                chart_specs = retry_parsed["charts"]
                            full_response += "\n" + retry_response
                    except Exception as e2:
                        st.error(f"SQL error: {err}")
                else:
                    # Auto-generate charts if LLM didn't provide any
                    if not chart_specs and df_result is not None:
                        chart_specs = auto_generate_charts(df_result)

                    # Render artifacts (dataframe + multiple ECharts)
                    msg_key = str(len(st.session_state.messages))
                    render_artifact(df_result, chart_specs, key_suffix=msg_key)

            # Handle pLTV scoring request
            if is_pltv_request and mode == "cfm" and df_result is None:
                models = list_available_models()
                if models:
                    selected_model = st.session_state.get("pltv_model_select", models[0])
                    with st.spinner(f"Running pLTV inference with {selected_model}…"):
                        scored_df, err = run_pltv_inference(selected_model,
                                                           max_rows=st.session_state.get("inference_max_rows", 50000))
                    if err:
                        st.error(err)
                    elif scored_df is not None:
                        scored_df = scored_df.sort_values("pltv_score", ascending=False)
                        top_pct = scored_df.head(max(1, len(scored_df) // 100))

                        st.markdown(f"**pLTV Scores** — {len(scored_df):,} users scored, "
                                    f"showing top 1% ({len(top_pct):,} users)")

                        pltv_charts = [
                            {"type": "bar", "x": "vopenid", "y": "pltv_score",
                             "title": "Top 1% Users — Predicted LTV30"},
                            {"type": "pie", "names": "media_source", "values": "pltv_score",
                             "title": "pLTV Score by Media Source"},
                        ]
                        msg_key = str(len(st.session_state.messages)) + "_pltv"
                        render_artifact(top_pct, pltv_charts, key_suffix=msg_key)

                        # Full scored table download
                        st.download_button(
                            "⬇️ Download full scored table",
                            scored_df.to_csv(index=False),
                            "cfm_pltv_scored.csv", "text/csv",
                            key=f"full_scored_{msg_key}",
                        )

            # Show citations
            if snippets:
                with st.expander("📚 Sources used"):
                    for s in snippets:
                        st.markdown(f"**[{s['file']} § {s['heading']}]** (relevance: {s['score']:.2f})")
                        st.caption(s["snippet"][:400] + ("…" if len(s["snippet"]) > 400 else ""))

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Store follow-ups for persistent rendering after rerun
        st.session_state.last_follow_ups = get_follow_up_suggestions(user_input)

    except ValueError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}"})
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})


# ---------------------------------------------------------------------------
# Sidebar: LLM settings + chatbot controls
# ---------------------------------------------------------------------------
def render_chatbot_sidebar():
    """Render chatbot-specific sidebar controls inside a collapsed expander."""
    from utils.llm_providers import PROVIDER_MODELS, PROVIDER_LABELS

    st.sidebar.markdown("---")

    with st.sidebar.expander("⚙️ Chatbot Settings", expanded=False):
        # Provider selection
        providers = list(PROVIDER_MODELS.keys())
        provider_idx = providers.index(st.session_state.get("llm_provider", "openai")) \
            if st.session_state.get("llm_provider", "openai") in providers else 0
        provider = st.selectbox(
            "LLM Provider",
            providers,
            index=provider_idx,
            format_func=lambda p: PROVIDER_LABELS.get(p, p),
            key="llm_provider",
        )

        # Model selection
        models = PROVIDER_MODELS.get(provider, [])
        model_idx = models.index(st.session_state.get("llm_model", models[0])) \
            if st.session_state.get("llm_model", models[0] if models else "") in models else 0
        st.selectbox(
            "Model",
            models,
            index=model_idx,
            key="llm_model",
        )

        # API key
        if "api_keys" not in st.session_state:
            st.session_state.api_keys = {}
        current_key = st.session_state.api_keys.get(provider, "")
        api_key = st.text_input(
            f"API Key ({PROVIDER_LABELS.get(provider, provider)})",
            value=current_key,
            type="password",
            key=f"api_key_input_{provider}",
        )
        st.session_state.api_keys[provider] = api_key

        st.markdown("---")

        # pLTV model selector
        available_models = list_available_models()
        if available_models:
            st.selectbox(
                "pLTV Model for scoring",
                available_models,
                index=0,
                key="pltv_model_select",
            )
            st.slider(
                "Max rows for inference",
                min_value=10000, max_value=200000, value=50000, step=10000,
                key="inference_max_rows",
            )

        st.markdown("---")

        # Schema viewer
        st.markdown("**📋 Dataset Schema**")
        st.code(COLUMN_SCHEMA, language="text")
        st.caption(f"Source: `{CSV_PATH.name}` — 37 columns")

    # Clear chat — keep outside expander for quick access
    if st.sidebar.button("🗑️ Clear chat", key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()

    # Session export
    if st.session_state.get("messages"):
        export_lines = []
        for msg in st.session_state.messages:
            role = "**User**" if msg["role"] == "user" else "**Assistant**"
            export_lines.append(f"### {role}\n\n{msg['content']}\n")
        export_md = "# CFM Chatbot — Session Export\n\n" + "\n---\n\n".join(export_lines)
        st.sidebar.download_button(
            "📥 Export conversation",
            export_md, "cfm_chatbot_session.md", "text/markdown",
            key="session_export_btn",
        )


# ---------------------------------------------------------------------------
# Starter question pills
# ---------------------------------------------------------------------------
STARTER_QUESTIONS = [
    ("�", "Give me the executive brief — key metrics overview"),
    ("�", "Show the Pareto concentration — top 1% revenue share"),
    ("💰", "Show me the late payer opportunity — how much revenue are we missing?"),
    ("📡", "Channel performance dashboard — compare all channels"),
    ("�", "Show cohort quality trends — is user quality declining?"),
    ("🧠", "Explain the model's feature importance"),
    ("🐋", "Show me whale concentration — top 1% vs rest"),
    ("🔍", "Detect anomalies and outliers in the data"),
    ("🎯", "Score all users with the pLTV model and show top 1%"),
    ("📉", "What is the data quality and coverage status?"),
    ("🌱", "Compare seed strategies: D7-payers-only vs enriched with late payers"),
    ("💡", "What does the causal inference analysis say about engagement driving conversion?"),
]


def render_starter_questions():
    """Render clickable starter question pills."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Try asking…")
    for emoji, question in STARTER_QUESTIONS:
        if st.sidebar.button(f"{emoji} {question[:50]}…" if len(question) > 50 else f"{emoji} {question}",
                             key=f"starter_{hash(question)}"):
            st.session_state.pending_question = question
            st.rerun()


# ---------------------------------------------------------------------------
# Question history helper
# ---------------------------------------------------------------------------
def get_user_questions() -> list[dict]:
    """Extract user questions from message history with their index."""
    questions = []
    for i, msg in enumerate(st.session_state.get("messages", [])):
        if msg["role"] == "user":
            questions.append({"index": i, "text": msg["content"]})
    return questions


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "cfm"
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}

    # Sidebar
    render_chatbot_sidebar()
    render_starter_questions()

    # Page header
    st.title("🤖 CFM Data Chatbot")
    st.caption(f"Ask questions about CrossFire Mobile pLTV data, whales, channels, revenue, and more. "
               f"Answers are grounded in `{CSV_PATH.name}` and 23 analysis reports. "
               f"Try the ⚡ instant analytics modules for Pareto, channels, late-payers, cohorts, and more.")

    # Check API key
    provider = st.session_state.get("llm_provider", "openai")
    api_key = st.session_state.get("api_keys", {}).get(provider, "")
    if not api_key:
        st.info(f"👋 Enter your **{provider.title()}** API key in the sidebar to start chatting.")

    # ── Fixed question-history panel CSS ──────────────────────────
    # Inject CSS that pins the question-history panel to the bottom-right
    # of the viewport so it remains visible as the user scrolls the chat.
    st.markdown("""
    <style>
    /* Fixed panel: bottom-right of viewport */
    .qh-fixed-panel {
        position: fixed;
        bottom: 120px;            /* above the chat_input bar */
        right: 24px;
        width: 26%;
        max-height: 45vh;
        overflow-y: auto;
        background: var(--background-color, #fff);
        border: 1px solid var(--secondary-background-color, #ddd);
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.08);
        z-index: 999;
        font-size: 0.88rem;
    }
    .qh-fixed-panel h4 { margin-top: 0; font-size: 0.95rem; }
    .qh-fixed-panel .qh-item {
        display: block;
        padding: 0.35rem 0.5rem;
        margin-bottom: 0.3rem;
        border-radius: 0.35rem;
        background: var(--secondary-background-color, #f0f2f6);
        color: inherit;
        text-decoration: none;
        cursor: pointer;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border: none;
        width: 100%;
        text-align: left;
        font-size: 0.85rem;
    }
    .qh-fixed-panel .qh-item:hover {
        filter: brightness(0.92);
    }
    .qh-empty { color: #999; font-size: 0.82rem; }
    /* Give the main content some right margin so chat doesn't go under the panel */
    section.main > div.block-container {
        padding-right: max(1rem, 28%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Build the question history HTML ───────────────────────────
    questions = get_user_questions()
    qh_items_html = ""
    if questions:
        for idx, q in enumerate(questions):
            q_label = q["text"][:55].replace("<", "&lt;").replace(">", "&gt;")
            if len(q["text"]) > 55:
                q_label += "…"
            msg_index = q["index"]
            qh_items_html += (
                f'<div class="qh-item" '
                f'onclick="document.getElementById(\'msg-anchor-{msg_index}\')?.scrollIntoView({{behavior:\'smooth\',block:\'center\'}})">'
                f'<b>Q{idx+1}.</b> {q_label}</div>\n'
            )
    else:
        qh_items_html = '<div class="qh-empty">Your questions will appear here as you chat.</div>'

    st.markdown(
        f'<div class="qh-fixed-panel"><h4>📝 Question History</h4>{qh_items_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Main chat area (full width) ───────────────────────────────
    # Mode indicator
    mode = st.session_state.get("chat_mode", "cfm")
    mode_label = "🎯 CFM Mode" if mode == "cfm" else "💬 General Mode"
    st.markdown(f"*Current mode: {mode_label}*")

    # Display conversation history
    for i, msg in enumerate(st.session_state.messages):
        # Insert a scroll-anchor before every message so question history can jump here
        st.markdown(f'<div id="msg-anchor-{i}"></div>', unsafe_allow_html=True)
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Render follow-up suggestions persistently after the last assistant message
    last_follow_ups = st.session_state.get("last_follow_ups", [])
    if last_follow_ups and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        render_follow_ups(last_follow_ups, key_prefix="persist_followup")

    # Handle pending question from follow-up buttons or starter pills
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask me about CFM payers, whales, revenue, pLTV…")

    if pending:
        user_input = pending

    if user_input:
        st.session_state.last_follow_ups = []  # clear while processing
        with st.chat_message("user"):
            st.markdown(user_input)
        handle_message(user_input)
        st.rerun()  # rerun so follow-up suggestions render immediately


main()
