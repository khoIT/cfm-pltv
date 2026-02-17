"""
Data Upload Module
Allows users to upload a CSV file and automatically splits it into:
- Training data (70%)
- Test 1 data (15%)
- Test 2 data (15%)
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def show_upload_interface():
    """Display file upload interface and handle data splitting."""
    st.markdown("### 📤 Upload Dataset")
    st.markdown(
        "Upload your full dataset CSV. It will be automatically split into:  \n"
        "- **Training data** (70%) → `cfm_pltv_train.csv`  \n"
        "- **Test 1 — OOT Near** (15%) → `cfm_pltv_test1.csv`  \n"
        "- **Test 2 — OOT Far** (15%) → `cfm_pltv_test2.csv`"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your complete dataset. Must contain columns: vopenid, ltv30, install_date, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Show file info
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"📁 File: **{uploaded_file.name}** ({file_size:.1f} MB)")
            
            # Load data
            with st.spinner("Reading CSV file..."):
                df = pd.read_csv(uploaded_file, low_memory=False)
            
            st.success(f"✅ Loaded **{len(df):,}** rows, **{len(df.columns)}** columns")
            
            # Show preview
            with st.expander("📊 Data Preview (first 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate required columns
            required_cols = ["vopenid", "ltv30", "install_date"]
            missing = [c for c in required_cols if c not in df.columns]
            
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.markdown("**Required columns:** `vopenid`, `ltv30`, `install_date`")
                return False
            
            # Show column info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                payer_rate = df["ltv30"].gt(0).mean() * 100 if "ltv30" in df.columns else 0
                st.metric("Payer Rate", f"{payer_rate:.1f}%")
            
            # Split configuration
            st.markdown("---")
            st.markdown("### ⚙️ Split Configuration")
            
            col_split1, col_split2, col_split3 = st.columns(3)
            with col_split1:
                train_pct = st.slider("Training %", 50, 90, 70, 5)
            with col_split2:
                test1_pct = st.slider("Test 1 %", 5, 30, 15, 5)
            with col_split3:
                test2_pct = 100 - train_pct - test1_pct
                st.metric("Test 2 %", f"{test2_pct}%", help="Automatically calculated")
            
            if test2_pct < 5:
                st.warning("⚠️ Test 2 percentage is too small. Adjust Training and Test 1 percentages.")
                return False
            
            # Show split preview
            n_train = int(len(df) * train_pct / 100)
            n_test1 = int(len(df) * test1_pct / 100)
            n_test2 = len(df) - n_train - n_test1
            
            st.markdown("**Split Preview:**")
            split_df = pd.DataFrame([
                {"Dataset": "Training", "Rows": f"{n_train:,}", "Percentage": f"{train_pct}%"},
                {"Dataset": "Test 1 (OOT Near)", "Rows": f"{n_test1:,}", "Percentage": f"{test1_pct}%"},
                {"Dataset": "Test 2 (OOT Far)", "Rows": f"{n_test2:,}", "Percentage": f"{test2_pct}%"},
            ])
            st.dataframe(split_df, width='stretch', hide_index=True)
            
            # Split strategy
            st.markdown("---")
            st.markdown("### 📅 Split Strategy")
            split_method = st.radio(
                "How to split the data?",
                ["Time-based (Recommended)", "Random"],
                help="Time-based: sorts by install_date and splits chronologically.  \n"
                     "Random: shuffles and splits randomly."
            )
            
            # Process button
            st.markdown("---")
            if st.button("🚀 Process & Save Datasets", type="primary", width='stretch'):
                with st.spinner("Processing and saving datasets..."):
                    success = process_and_save(
                        df, train_pct, test1_pct, test2_pct,
                        time_based=(split_method == "Time-based (Recommended)")
                    )
                    
                    if success:
                        st.success("✅ **Datasets saved successfully!**")
                        st.balloons()
                        st.markdown(
                            "**Files created:**  \n"
                            f"- `data/cfm_pltv_train.csv` ({n_train:,} rows)  \n"
                            f"- `data/cfm_pltv_test1.csv` ({n_test1:,} rows)  \n"
                            f"- `data/cfm_pltv_test2.csv` ({n_test2:,} rows)  \n\n"
                            "**Next step:** Refresh the page to start using your data!"
                        )
                        
                        # Add rerun button
                        if st.button("🔄 Reload App with New Data"):
                            st.rerun()
                        
                        return True
                    else:
                        st.error("❌ Failed to save datasets. Check error messages above.")
                        return False
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return False
    
    return False


def process_and_save(df: pd.DataFrame, train_pct: int, test1_pct: int, test2_pct: int, time_based: bool = True) -> bool:
    """Split data and save to CSV files. Always saves as cfm_pltv_train.csv regardless of original filename."""
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sort or shuffle
        if time_based:
            if "install_date" not in df.columns:
                st.error("❌ install_date column not found. Cannot use time-based split.")
                return False
            df = df.sort_values("install_date").reset_index(drop=True)
            st.info("📅 Sorted by install_date (chronological split)")
        else:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            st.info("🔀 Shuffled randomly")
        
        # Calculate split indices
        n_train = int(len(df) * train_pct / 100)
        n_test1 = int(len(df) * test1_pct / 100)
        
        # Split
        df_train = df.iloc[:n_train]
        df_test1 = df.iloc[n_train:n_train + n_test1]
        df_test2 = df.iloc[n_train + n_test1:]
        
        # Save with standardized filenames (regardless of original upload name)
        train_path = DATA_DIR / "cfm_pltv_train.csv"
        test1_path = DATA_DIR / "cfm_pltv_test1.csv"
        test2_path = DATA_DIR / "cfm_pltv_test2.csv"
        
        st.info(f"💾 Saving as: cfm_pltv_train.csv, cfm_pltv_test1.csv, cfm_pltv_test2.csv")
        
        df_train.to_csv(train_path, index=False)
        df_test1.to_csv(test1_path, index=False)
        df_test2.to_csv(test2_path, index=False)
        
        # Create metadata file
        metadata = {
            "upload_timestamp": datetime.now().isoformat(),
            "original_file": "user_upload",
            "total_rows": len(df),
            "train_rows": len(df_train),
            "test1_rows": len(df_test1),
            "test2_rows": len(df_test2),
            "split_method": "time_based" if time_based else "random",
            "train_pct": train_pct,
            "test1_pct": test1_pct,
            "test2_pct": test2_pct,
        }
        
        metadata_path = DATA_DIR / "upload_metadata.txt"
        with open(metadata_path, "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        return False


def check_data_exists() -> bool:
    """Check if training data exists."""
    train_path = DATA_DIR / "cfm_pltv_train.csv"
    return train_path.exists()
