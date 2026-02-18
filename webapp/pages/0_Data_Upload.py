"""
Page 0 — Data Upload
Upload and split dataset into training and test sets.
Only shown when data directory doesn't exist or is empty.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import render_sidebar, render_top_menu, DATA_DIR
from data_uploader import show_upload_interface, check_data_exists

render_top_menu()
render_sidebar()

st.title("📤 Data Upload")

# Check if data already exists
if check_data_exists():
    st.success("✅ Training data already exists!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", "Ready to use")
    with col2:
        train_path = DATA_DIR / "cfm_pltv_train.csv"
        if train_path.exists():
            import os
            size_mb = os.path.getsize(train_path) / (1024 * 1024)
            st.metric("Training Data Size", f"{size_mb:.1f} MB")
    
    st.markdown("---")
    st.info(
        "💡 **Your data is ready!** Navigate to other pages to start analysis.  \n\n"
        "If you want to upload a new dataset, expand the section below."
    )
    
    with st.expander("🔄 Upload New Dataset (Replace Existing)", expanded=False):
        st.warning(
            "⚠️ **Warning:** Uploading a new dataset will **overwrite** existing files:  \n"
            "- `cfm_pltv_train.csv`  \n"
            "- `cfm_pltv_test1.csv`  \n"
            "- `cfm_pltv_test2.csv`  \n\n"
            "Any trained models in session will be lost."
        )
        
        if st.checkbox("I understand, proceed with upload"):
            show_upload_interface()
else:
    st.info(
        "👋 **Welcome!** No training data found.  \n\n"
        "Upload your dataset below to get started. The file will be automatically split into:  \n"
        "- **Training set** (70%) — for model training  \n"
        "- **Test 1 — OOT Near** (15%) — near-term validation  \n"
        "- **Test 2 — OOT Far** (15%) — long-term validation"
    )
    
    st.markdown("---")
    show_upload_interface()

# Show existing files if any
st.markdown("---")
st.markdown("### 📁 Current Data Files")

files_info = []
for fname in ["cfm_pltv_train.csv", "cfm_pltv_test1.csv", "cfm_pltv_test2.csv"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        import os
        import pandas as pd
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        try:
            nrows = len(pd.read_csv(fpath, usecols=[0]))
            files_info.append({
                "File": fname,
                "Size (MB)": f"{size_mb:.1f}",
                "Rows": f"{nrows:,}",
                "Status": "✅ Ready"
            })
        except:
            files_info.append({
                "File": fname,
                "Size (MB)": f"{size_mb:.1f}",
                "Rows": "Error",
                "Status": "⚠️ Check file"
            })
    else:
        files_info.append({
            "File": fname,
            "Size (MB)": "-",
            "Rows": "-",
            "Status": "❌ Missing"
        })

if files_info:
    import pandas as pd
    st.dataframe(pd.DataFrame(files_info), width=1000, hide_index=True)

# Show metadata if exists
metadata_path = DATA_DIR / "upload_metadata.txt"
if metadata_path.exists():
    with st.expander("📋 Upload Metadata", expanded=False):
        st.code(metadata_path.read_text(), language=None)
