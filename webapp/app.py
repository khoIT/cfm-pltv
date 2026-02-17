"""
CFM Decision Intelligence — Streamlit App Entry Point
Showcases the Decision-Centric Intelligence Loop for pLTV / UA Seed Optimization.
"""
import streamlit as st
from pathlib import Path
import sys

st.set_page_config(
    page_title="CFM Decision Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make shared importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import render_sidebar, get_data, REPORTS_DIR

render_sidebar()


def read_md(filename: str) -> str:
    path = REPORTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"> ⚠️ Report `{filename}` not found."


# --- Main Page: Summary ---
st.title("CFM Decision Intelligence — Summary")

# Check if data is missing before trying to load
if st.session_state.get("data_missing", False):
    st.warning("⚠️ **No training data found**")
    st.info(
        "👋 **Welcome to CFM Decision Intelligence!**  \n\n"
        "To get started, please upload your dataset:  \n"
        "1. Navigate to **📤 Data Upload** page in the sidebar  \n"
        "2. Upload your CSV file  \n"
        "3. The system will automatically split it into training and test sets  \n"
        "4. Return here to see the full analysis"
    )
    st.markdown("---")
    st.markdown("### 📊 What This App Does")
    st.markdown(
        "This application implements a **Decision-Centric Intelligence Loop** for:  \n"
        "- **pLTV Prediction** — Predict 30-day Lifetime Value of users  \n"
        "- **UA Seed Optimization** — Select high-value users for lookalike targeting  \n"
        "- **Whale Analysis** — Understand revenue concentration and whale behavior  \n"
        "- **Model Evaluation** — Compare ML models vs simple heuristics  \n"
        "- **Action Simulation** — Simulate seed selection strategies and ROI"
    )
    st.stop()

df = get_data()
st.caption(f"Training data: **{len(df):,}** rows (2025-12-16 to 2026-01-08)")
st.markdown("---")

summary_md = read_md("CFM_Decision_Intelligence_Summary.md")

# Replace markdown links with expanders to avoid navigation errors
import re

# Extract report references and create expanders instead of links
lines = summary_md.split('\n')
processed_lines = []
for line in lines:
    # Match pattern: See → [filename.md](filename.md)
    if 'See →' in line and '.md](' in line:
        # Extract the filename
        match = re.search(r'\[([^\]]+\.md)\]', line)
        if match:
            filename = match.group(1)
            # Replace with expander reference
            line = line.split('See →')[0] + f"_See report in sidebar: {filename}_"
    processed_lines.append(line)

st.markdown('\n'.join(processed_lines))

st.markdown("---")
st.info("👈 Use the sidebar to toggle **currency (USD/VND)**, adjust **row limits**, then navigate through each layer.")
