"""Quick smoke test for chatbot components (run from webapp/ directory)."""
import sys, re, json, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # cfm-pltv/
DATA_DIR  = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # webapp/

PASS = "✅"
FAIL = "❌"

# ---------------------------------------------------------------------------
# 1. DuckDB + CSV
# ---------------------------------------------------------------------------
print("── 1. DuckDB ─────────────────────────────────────")
import duckdb
print(f"   version : {duckdb.__version__}")

# Auto-detect CSV (same logic as CFM_Chatbot.py)
def _find_csv():
    for name in ("cfm_pltv_Feb22.csv", "cfm_pltv_train.csv"):
        p = DATA_DIR / name
        if p.exists():
            return p
    globbed = sorted(DATA_DIR.glob("cfm_pltv*.csv"))
    return globbed[0] if globbed else None

csv_path = _find_csv()
assert csv_path, f"{FAIL} No cfm_pltv*.csv found in {DATA_DIR}"
print(f"   csv     : {csv_path.name}")

con = duckdb.connect()
con.execute(f"CREATE VIEW cfm_features AS SELECT * FROM read_csv_auto('{csv_path}')")
r = con.execute("SELECT COUNT(*) AS n FROM cfm_features").fetchdf()
cols = con.execute("SELECT * FROM cfm_features LIMIT 1").fetchdf().columns.tolist()
print(f"   rows    : {int(r['n'].iloc[0]):,}")
print(f"   columns : {len(cols)} → {cols}")

REQUIRED_COLS = {"ltv30","is_payer_30","rev_d7","media_source","install_date",
                 "active_days_d7","games_d7","kills_d7","first_os","vopenid"}
missing = REQUIRED_COLS - set(cols)
assert not missing, f"{FAIL} Missing columns: {missing}"
print(f"   {PASS} All required columns present")

# ---------------------------------------------------------------------------
# 2. Analytics SQL smoke tests
# ---------------------------------------------------------------------------
print("\n── 2. Analytics SQL ──────────────────────────────")

def exec_sql(sql, limit=100):
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + f" LIMIT {limit}"
    try:
        return con.execute(sql).fetchdf(), ""
    except Exception as e:
        return None, str(e)

SQL_TESTS = {
    "pareto":    "SELECT ltv30, is_payer_30 FROM cfm_features WHERE ltv30 > 0 ORDER BY ltv30 DESC",
    "late_payer":"SELECT CASE WHEN rev_d7>0 THEN 'D7' WHEN ltv30>0 THEN 'Late' ELSE 'Non' END AS seg, COUNT(*) AS n FROM cfm_features GROUP BY seg",
    "channel":   "SELECT media_source, COUNT(*) AS users, AVG(ltv30) AS arpu FROM cfm_features WHERE media_source IS NOT NULL AND media_source!='' GROUP BY media_source HAVING COUNT(*)>=1000 ORDER BY arpu DESC",
    "cohort":    "SELECT DATE_TRUNC('week', CAST(install_date AS DATE)) AS wk, COUNT(*) AS n FROM cfm_features WHERE install_date IS NOT NULL GROUP BY wk HAVING COUNT(*)>=100 ORDER BY wk",
    "exec_brief":"SELECT COUNT(*) AS total, SUM(ltv30) AS rev, AVG(ltv30) AS arpu, 100.0*SUM(CASE WHEN is_payer_30=1 THEN 1 ELSE 0 END)/COUNT(*) AS payer_rate FROM cfm_features",
    "anomaly":   "SELECT vopenid, ltv30, rev_d7, active_days_d7, games_d7 FROM cfm_features ORDER BY ltv30 DESC",
    "data_qual": "SELECT COUNT(*) AS n, COUNT(DISTINCT media_source) AS channels, COUNT(DISTINCT first_os) AS os_count, MIN(install_date) AS first_date, MAX(install_date) AS last_date FROM cfm_features",
    "median":    "SELECT MEDIAN(ltv30) AS med FROM cfm_features WHERE ltv30>0",
}

all_ok = True
for name, sql in SQL_TESTS.items():
    df, err = exec_sql(sql)
    status = PASS if df is not None and not df.empty else FAIL
    if status == FAIL:
        all_ok = False
    print(f"   {status} {name:<12} rows={len(df) if df is not None else 0}  {err or ''}")

assert all_ok, f"{FAIL} Some SQL tests failed"

# ---------------------------------------------------------------------------
# 3. Deterministic flow keyword detection
# ---------------------------------------------------------------------------
print("\n── 3. Flow detection ─────────────────────────────")
FLOW_KEYWORDS = {
    "pareto":           ["pareto", "top 1%", "whale concentration", "80/20"],
    "late_payer":       ["late payer", "late-payer", "d8", "missed revenue"],
    "channel":          ["channel", "media_source", "channel dashboard", "channel performance"],
    "cohort":           ["cohort", "weekly cohort", "install week", "quality declining"],
    "feature_importance":["feature importance", "xgboost", "model feature", "what predicts"],
    "executive_brief":  ["executive", "exec brief", "summary of all", "key metrics"],
    "anomaly":          ["anomaly", "outlier", "mega-whale", "unusual"],
    "data_quality":     ["data quality", "null rate", "coverage", "completeness"],
}

def detect_flow(text):
    t = text.lower()
    for flow, kws in FLOW_KEYWORDS.items():
        if any(k in t for k in kws):
            return flow
    return None

FLOW_TESTS = [
    ("Show me the pareto curve",                  "pareto"),
    ("How much late payer revenue are we losing", "late_payer"),
    ("Channel dashboard please",                  "channel"),
    ("Is cohort quality declining?",              "cohort"),
    ("What are the feature importance scores?",   "feature_importance"),
    ("Give me the executive brief",               "executive_brief"),
    ("Show anomaly detection",                    "anomaly"),
    ("Run data quality check",                    "data_quality"),
    ("What is the weather today?",                None),
]

flow_ok = True
for question, expected in FLOW_TESTS:
    got = detect_flow(question)
    ok = got == expected
    if not ok:
        flow_ok = False
    print(f"   {'✅' if ok else '❌'} '{question[:45]}' → {got} (expected {expected})")

assert flow_ok, f"{FAIL} Flow detection mismatches"

# ---------------------------------------------------------------------------
# 4. Report index (TF-IDF)
# ---------------------------------------------------------------------------
print("\n── 4. Report index ───────────────────────────────")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

md_files = sorted(REPORTS_DIR.glob("*.md"))
chunks = []
for md_path in md_files:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r'^(##\s+.+)$', text, flags=re.MULTILINE)
    heading, buf = "Introduction", ""
    for part in sections:
        if part.startswith("## "):
            if buf.strip():
                chunks.append({"file": md_path.name, "heading": heading, "text": buf.strip()[:3000]})
            heading, buf = part.strip("# ").strip(), ""
        else:
            buf += part + "\n"
    if buf.strip():
        chunks.append({"file": md_path.name, "heading": heading, "text": buf.strip()[:3000]})

print(f"   files   : {len(md_files)}")
print(f"   chunks  : {len(chunks)}")
assert len(chunks) > 0, f"{FAIL} No report chunks found — check REPORTS_DIR={REPORTS_DIR}"

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform([c["text"] for c in chunks])
q_vec = vectorizer.transform(["whale concentration revenue"])
scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
top_idx = scores.argsort()[::-1][:3]
print("   Top 3 for 'whale concentration revenue':")
for i in top_idx:
    print(f"     [{chunks[i]['file'][:40]} § {chunks[i]['heading'][:30]}] score={scores[i]:.3f}")
print(f"   {PASS} TF-IDF index built")

# ---------------------------------------------------------------------------
# 5. Model loading
# ---------------------------------------------------------------------------
print("\n── 5. Model loading ──────────────────────────────")
model_dir = None
for p in sorted(MODELS_DIR.iterdir()):
    if p.is_dir() and (p / "metadata.json").exists():
        model_dir = p
        break

assert model_dir, f"{FAIL} No model directory with metadata.json found in {MODELS_DIR}"
meta = json.loads((model_dir / "metadata.json").read_text())
print(f"   dir     : {model_dir.name}")
print(f"   type    : {meta.get('model_type', 'unknown')}")
print(f"   features: {meta.get('n_features', '?')} → {meta.get('features', [])[:5]}...")

model_pkl = model_dir / "model.pkl"
if model_pkl.exists():
    try:
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        has_imp = hasattr(model, "feature_importances_")
        print(f"   {PASS} model.pkl loaded  feature_importances_={has_imp}")
    except Exception as e:
        print(f"   ⚠️  model.pkl load error (non-fatal): {e}")
else:
    print(f"   ⚠️  model.pkl not found (feature importance will use metadata fallback)")

fi = meta.get("feature_importances", {})
print(f"   importances in metadata: {len(fi)} features  top={max(fi, key=fi.get) if fi else 'n/a'}")

# ---------------------------------------------------------------------------
# 6. LLM providers import
# ---------------------------------------------------------------------------
print("\n── 6. LLM providers ──────────────────────────────")
try:
    from utils.llm_providers import PROVIDER_MODELS, PROVIDER_LABELS
    print(f"   {PASS} providers: {list(PROVIDER_MODELS.keys())}")
except ImportError as e:
    print(f"   ⚠️  llm_providers import error: {e}")

# ---------------------------------------------------------------------------
print("\n" + "═"*50)
print("✅ All chatbot smoke tests passed")
print("═"*50)
