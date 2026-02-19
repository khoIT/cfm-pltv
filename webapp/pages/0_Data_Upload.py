"""
Page 0 — Data Upload
Upload and split dataset into training and test sets.
Manage registered datasets (rename, delete).
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import render_sidebar, render_top_menu, DATA_DIR
from data_uploader import show_upload_interface, check_data_exists, show_dataset_management

render_top_menu()
render_sidebar()

st.title("📤 Data Upload & Management")

# ── Tab layout: Upload vs Manage ────────────────────────────────
tab_upload, tab_manage = st.tabs(["� Upload New", "📚 Manage Datasets"])

with tab_upload:
    if check_data_exists():
        st.success("✅ Datasets already registered! You can upload more below.")
        st.markdown("---")

    show_upload_interface()

with tab_manage:
    show_dataset_management()

    # Also show raw file list for transparency
    st.markdown("---")
    st.markdown("### 📁 All Data Files")
    import os
    import pandas as pd
    files_info = []
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.glob("*.csv")):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            files_info.append({
                "File": f.name,
                "Size (MB)": f"{size_mb:.1f}",
                "Status": "✅",
            })
    if files_info:
        st.dataframe(pd.DataFrame(files_info), use_container_width=True, hide_index=True)
    else:
        st.info("No CSV files found in data/ directory.")
