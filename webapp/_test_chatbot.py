"""Quick smoke test for chatbot components."""
import sys
sys.path.insert(0, ".")

# 1. DuckDB
import duckdb
print("DuckDB version:", duckdb.__version__)
con = duckdb.connect()
con.execute("CREATE VIEW cfm_features AS SELECT * FROM read_csv_auto('e:/code/cfm_pltv/data/cfm_pltv_Feb22.csv')")
r = con.execute("SELECT COUNT(*) as cnt FROM cfm_features").fetchdf()
print("Rows:", r["cnt"][0])
cols = con.execute("DESCRIBE cfm_features").fetchdf()
print("Columns:", len(cols))
print(cols["column_name"].tolist())

# 2. Report retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

REPORTS_DIR = Path("e:/code/cfm_pltv/reports")
chunks = []
for md_path in sorted(REPORTS_DIR.glob("*.md")):
    text = md_path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r'^(##\s+.+)$', text, flags=re.MULTILINE)
    current_heading = "Introduction"
    buf = ""
    for part in sections:
        if part.startswith("## "):
            if buf.strip():
                chunks.append({"file": md_path.name, "heading": current_heading, "text": buf.strip()[:3000]})
            current_heading = part.strip("# ").strip()
            buf = ""
        else:
            buf += part + "\n"
    if buf.strip():
        chunks.append({"file": md_path.name, "heading": current_heading, "text": buf.strip()[:3000]})

print(f"\nReport index: {len(chunks)} chunks from {len(list(REPORTS_DIR.glob('*.md')))} files")

corpus = [c["text"] for c in chunks]
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(corpus)

q_vec = vectorizer.transform(["whale concentration revenue"])
scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
top_idx = scores.argsort()[::-1][:3]
print("Top 3 for 'whale concentration revenue':")
for i in top_idx:
    print(f"  [{chunks[i]['file']} § {chunks[i]['heading']}] score={scores[i]:.3f}")

# 3. Model loading
import pickle, json
model_dir = Path("e:/code/cfm_pltv/models/pltv_model_20260223_14_16Mrows")
with open(model_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)
meta = json.loads((model_dir / "metadata.json").read_text())
print(f"\nModel loaded: {meta['model_name']}, {meta['n_features']} features")
print("Features:", meta["features"])

# 4. LLM providers import
from utils.llm_providers import PROVIDER_MODELS, PROVIDER_LABELS
print(f"\nLLM providers: {list(PROVIDER_MODELS.keys())}")

print("\n✅ All chatbot components OK")
