"""
Data Upload Module
Allows users to upload a CSV file and split it into 1, 2, or 3 datasets.
Each dataset is registered in the Dataset Registry and can be used independently per page.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def show_upload_interface():
    """Display file upload interface with flexible split options."""
    st.markdown("### 📤 Upload Dataset")
    st.markdown(
        "Upload your CSV file. You can choose to keep it as **one dataset** "
        "or **split** it into 2 or 3 parts. Each part is registered separately "
        "and can be used independently on any page."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your complete dataset. Must contain columns: vopenid, ltv30, install_date, etc."
    )

    if uploaded_file is not None:
        try:
            file_size = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 File: **{uploaded_file.name}** ({file_size:.1f} MB)")

            with st.spinner("Reading CSV file..."):
                df = pd.read_csv(uploaded_file, low_memory=False)

            st.success(f"✅ Loaded **{len(df):,}** rows, **{len(df.columns)}** columns")

            with st.expander("📊 Data Preview (first 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Validate required columns
            required_cols = ["vopenid", "ltv30", "install_date"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                return False

            # Column info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                payer_rate = df["ltv30"].gt(0).mean() * 100 if "ltv30" in df.columns else 0
                st.metric("Payer Rate", f"{payer_rate:.1f}%")

            # ── Split Configuration ────────────────────────────────
            st.markdown("---")
            st.markdown("### ⚙️ Split Configuration")

            num_splits = st.radio(
                "How many datasets?",
                [1, 2, 3],
                index=2,
                horizontal=True,
                help="**1**: Keep the file as a single dataset.  \n"
                     "**2**: Split into two (e.g., Train + Test).  \n"
                     "**3**: Split into three (e.g., Train + Test1 + Test2).",
            )

            # Dataset names
            st.markdown("#### 📝 Name your datasets")
            base_name = Path(uploaded_file.name).stem.replace(" ", "_")

            if num_splits == 1:
                name_1 = st.text_input("Dataset name", value=base_name, key="ds_name_1")
                pct_1 = 100
                pct_2 = pct_3 = 0
            elif num_splits == 2:
                nc1, nc2 = st.columns(2)
                with nc1:
                    name_1 = st.text_input("Dataset 1 name", value=f"{base_name}_train", key="ds_name_1")
                with nc2:
                    name_2 = st.text_input("Dataset 2 name", value=f"{base_name}_test", key="ds_name_2")
                sc1, sc2 = st.columns(2)
                with sc1:
                    pct_1 = st.slider("Dataset 1 %", 10, 90, 70, 5, key="pct_1")
                with sc2:
                    pct_2 = 100 - pct_1
                    st.metric("Dataset 2 %", f"{pct_2}%")
                pct_3 = 0
            else:  # 3
                nc1, nc2, nc3 = st.columns(3)
                with nc1:
                    name_1 = st.text_input("Dataset 1 name", value=f"{base_name}_train", key="ds_name_1")
                with nc2:
                    name_2 = st.text_input("Dataset 2 name", value=f"{base_name}_test1", key="ds_name_2")
                with nc3:
                    name_3 = st.text_input("Dataset 3 name", value=f"{base_name}_test2", key="ds_name_3")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    pct_1 = st.slider("Dataset 1 %", 10, 80, 70, 5, key="pct_1")
                with sc2:
                    pct_2 = st.slider("Dataset 2 %", 5, 40, 15, 5, key="pct_2")
                with sc3:
                    pct_3 = 100 - pct_1 - pct_2
                    st.metric("Dataset 3 %", f"{pct_3}%")
                if pct_3 < 5:
                    st.warning("⚠️ Dataset 3 is too small. Adjust the other sliders.")
                    return False

            # Split preview
            n1 = int(len(df) * pct_1 / 100) if num_splits > 1 else len(df)
            n2 = int(len(df) * pct_2 / 100) if num_splits >= 2 else 0
            n3 = len(df) - n1 - n2 if num_splits >= 3 else 0

            preview_rows = [{"Dataset": name_1, "Rows": f"{n1:,}", "%": f"{pct_1}%"}]
            if num_splits >= 2:
                preview_rows.append({"Dataset": name_2, "Rows": f"{n2:,}", "%": f"{pct_2}%"})
            if num_splits >= 3:
                preview_rows.append({"Dataset": name_3, "Rows": f"{n3:,}", "%": f"{pct_3}%"})
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            # Split strategy
            st.markdown("---")
            split_method = st.radio(
                "Split strategy",
                ["Time-based (Recommended)", "Random"],
                help="Time-based: sorts by install_date and splits chronologically.  \n"
                     "Random: shuffles and splits randomly.",
            ) if num_splits > 1 else "No split"

            # Process
            st.markdown("---")
            if st.button("🚀 Process & Save Datasets", type="primary", use_container_width=True):
                with st.spinner("Processing and saving datasets..."):
                    names = [name_1]
                    pcts = [pct_1]
                    if num_splits >= 2:
                        names.append(name_2)
                        pcts.append(pct_2)
                    if num_splits >= 3:
                        names.append(name_3)
                        pcts.append(pct_3)

                    success = process_and_save_flexible(
                        df, names, pcts,
                        time_based=(split_method == "Time-based (Recommended)"),
                        source_file=uploaded_file.name,
                    )
                    if success:
                        st.success("✅ **Datasets saved and registered!**")
                        st.balloons()
                        st.markdown("**Next step:** Select a dataset from the sidebar on any page.")
                        if st.button("🔄 Reload App"):
                            st.rerun()
                        return True
                    else:
                        st.error("❌ Failed to save. Check errors above.")
                        return False

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return False

    return False


def process_and_save_flexible(df: pd.DataFrame, names: list, pcts: list,
                               time_based: bool = True, source_file: str = "") -> bool:
    """Split data into N datasets, save to CSV, and register each."""
    from dataset_registry import register_dataset

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Sort or shuffle
        if time_based and len(names) > 1:
            if "install_date" not in df.columns:
                st.error("❌ install_date column not found. Cannot use time-based split.")
                return False
            df = df.sort_values("install_date").reset_index(drop=True)
        elif len(names) > 1:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split
        splits = []
        start = 0
        for i, (name, pct) in enumerate(zip(names, pcts)):
            if i == len(names) - 1:
                chunk = df.iloc[start:]
            else:
                n = int(len(df) * pct / 100)
                chunk = df.iloc[start:start + n]
                start += n
            splits.append((name, chunk))

        # Save and register
        created = []
        for name, chunk in splits:
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            filename = f"{safe_name}.csv"
            filepath = DATA_DIR / filename
            chunk.to_csv(filepath, index=False)

            split_info = f"{len(chunk):,} rows ({len(chunk)/len(df)*100:.0f}%)"
            ds_id = register_dataset(
                name=name,
                filename=filename,
                rows=len(chunk),
                columns=len(chunk.columns),
                source_file=source_file,
                split_info=split_info,
            )
            created.append((ds_id, name, len(chunk), filename))

        # Display created datasets
        for ds_id, name, nrows, fname in created:
            st.markdown(f"- ✅ **{name}** → `{fname}` ({nrows:,} rows)")

        # Also save legacy files for backward compat if 3-way split
        if len(names) == 3:
            _save_legacy_aliases(splits)

        return True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False


def _save_legacy_aliases(splits):
    """Save legacy cfm_pltv_train/test1/test2 aliases for backward compatibility."""
    legacy_map = [
        ("cfm_pltv_train.csv", 0),
        ("cfm_pltv_test1.csv", 1),
        ("cfm_pltv_test2.csv", 2),
    ]
    for fname, idx in legacy_map:
        if idx < len(splits):
            path = DATA_DIR / fname
            splits[idx][1].to_csv(path, index=False)


def check_data_exists() -> bool:
    """Check if any datasets exist (registry or legacy)."""
    from dataset_registry import list_datasets
    datasets = list_datasets()
    if datasets:
        return True
    train_path = DATA_DIR / "cfm_pltv_train.csv"
    return train_path.exists()


def show_dataset_management():
    """Display dataset management UI for renaming and deleting."""
    from dataset_registry import list_datasets, delete_dataset, rename_dataset, get_registry, PAGE_IDS

    datasets = list_datasets()
    if not datasets:
        st.info("No datasets registered. Upload a file above.")
        return

    st.markdown("### 📚 Manage Datasets")
    reg = get_registry()

    for ds_id, meta in datasets.items():
        pages_using = [PAGE_IDS.get(p, p) for p, d in reg.get("page_bindings", {}).items() if d == ds_id]
        pages_str = ", ".join(pages_using) if pages_using else "none"

        with st.expander(f"📄 {meta['name']} — {meta['rows']:,} rows ({meta['size_mb']} MB)", expanded=False):
            st.markdown(f"- **File:** `{meta['filename']}`")
            st.markdown(f"- **Created:** {meta.get('created_at', 'unknown')}")
            st.markdown(f"- **Source:** {meta.get('source_file', 'unknown')}")
            st.markdown(f"- **Used by pages:** {pages_str}")

            col_r, col_d = st.columns(2)
            with col_r:
                new_name = st.text_input("Rename", value=meta["name"], key=f"ren_{ds_id}")
                if new_name != meta["name"]:
                    if st.button("✏️ Rename", key=f"ren_btn_{ds_id}"):
                        rename_dataset(ds_id, new_name)
                        st.success(f"Renamed to: {new_name}")
                        st.rerun()
            with col_d:
                if pages_using:
                    st.warning(f"Used by {len(pages_using)} page(s)")
                if st.button("🗑️ Delete", key=f"del_btn_{ds_id}", type="secondary"):
                    if st.session_state.get(f"confirm_del_{ds_id}"):
                        delete_dataset(ds_id)
                        st.success(f"Deleted: {meta['name']}")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_del_{ds_id}"] = True
                        st.warning("Click **Delete** again to confirm.")
                        st.rerun()
