"""
Data Upload Module
Uploads a full dataset dump and automatically splits it into:
  - cfm_pltv_train  : mature users (install_date <= dump_date - 30d), 80% random split
  - cfm_pltv_test   : mature users (install_date <= dump_date - 30d), 20% random split
  - cfm_pltv_recent : recent users (install_date > dump_date - 30d), LTV30 not yet realized

The dump_date is inferred as the maximum install_date in the file (or overridable by the user).
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MATURITY_DAYS = 30          # users need this many days for LTV30 to be realized
TRAIN_FRACTION = 0.80       # fraction of mature users assigned to train


def _infer_dump_date(df: pd.DataFrame) -> pd.Timestamp:
    """Return the latest install_date in the dataframe as the effective dump date."""
    return pd.to_datetime(df["install_date"]).max()


def _split_by_maturity(df: pd.DataFrame, dump_date: pd.Timestamp,
                        maturity_days: int, train_frac: float, random_seed: int = 42):
    """
    Split df into (mature_train, mature_test, recent).
    mature = install_date <= dump_date - maturity_days
    recent = install_date >  dump_date - maturity_days
    """
    df = df.copy()
    df["_install_dt"] = pd.to_datetime(df["install_date"])
    cutoff = dump_date - timedelta(days=maturity_days)

    mature = df[df["_install_dt"] <= cutoff].drop(columns=["_install_dt"])
    recent = df[df["_install_dt"] >  cutoff].drop(columns=["_install_dt"])

    # Random 80/20 split within mature cohort
    mature_shuffled = mature.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_train = int(len(mature_shuffled) * train_frac)
    train = mature_shuffled.iloc[:n_train].reset_index(drop=True)
    test  = mature_shuffled.iloc[n_train:].reset_index(drop=True)

    return train, test, recent, cutoff


def show_upload_interface():
    """Display file upload interface with automatic maturity-based split."""
    st.markdown("### 📤 Upload Dataset")
    st.markdown(
        "Upload your **full dataset dump** (all users, Dec 16 → today). "
        "The system will automatically split it into three datasets based on data maturity:"
    )
    st.info(
        "- **Train** — users installed ≥30 days ago (LTV30 fully realized), 80% random split  \n"
        "- **Test** — same mature cohort, remaining 20% holdout  \n"
        "- **Recent** — users installed <30 days ago (LTV30 not yet realized — for live scoring)",
        icon="📊"
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Must contain columns: vopenid, ltv30, install_date"
    )

    if uploaded_file is None:
        return False

    try:
        file_size = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 **{uploaded_file.name}** — {file_size:.1f} MB")

        with st.spinner("Reading CSV…"):
            df = pd.read_csv(uploaded_file, low_memory=False)

        st.success(f"✅ Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")

        with st.expander("📊 Data Preview (first 10 rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        # Validate required columns
        required_cols = ["vopenid", "ltv30", "install_date"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            return False

        df["install_date"] = pd.to_datetime(df["install_date"])
        inferred_dump_date = _infer_dump_date(df)
        min_date = df["install_date"].min()
        payer_rate = df["ltv30"].gt(0).mean() * 100

        # ── Dataset overview ──────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Rows", f"{len(df):,}")
        with c2: st.metric("Date Range", f"{min_date.date()} → {inferred_dump_date.date()}")
        with c3: st.metric("Payer Rate", f"{payer_rate:.1f}%")
        with c4: st.metric("Columns", len(df.columns))

        # ── Split configuration ───────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚙️ Split Configuration")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            maturity_days = st.number_input(
                "Maturity threshold (days)",
                min_value=7, max_value=90, value=MATURITY_DAYS, step=1,
                help="Users installed more than this many days before the dump date "
                     "have a fully-realized LTV30 and go into Train/Test. "
                     "More recent users go into Recent."
            )
        with col_cfg2:
            train_pct = st.slider(
                "Train % (of mature users)",
                min_value=50, max_value=95, value=int(TRAIN_FRACTION * 100), step=5,
                help="Fraction of mature users assigned to Train. Remainder goes to Test."
            )
        with col_cfg3:
            dump_date_override = st.date_input(
                "Dump date (auto-detected)",
                value=inferred_dump_date.date(),
                help="The reference date used to compute maturity. "
                     "Defaults to the latest install_date in the file."
            )

        dump_date = pd.Timestamp(dump_date_override)
        train_frac = train_pct / 100.0

        # ── Preview split ─────────────────────────────────────────
        train_df, test_df, recent_df, cutoff = _split_by_maturity(
            df, dump_date, int(maturity_days), train_frac)

        preview = pd.DataFrame([
            {
                "Dataset": "cfm_pltv_train",
                "Users": f"{len(train_df):,}",
                "% of total": f"{len(train_df)/len(df)*100:.1f}%",
                "Install date range": f"≤ {cutoff.date()}",
                "LTV30 realized?": "✅ Yes",
                "Use for": "Model training, all analysis pages",
            },
            {
                "Dataset": "cfm_pltv_test",
                "Users": f"{len(test_df):,}",
                "% of total": f"{len(test_df)/len(df)*100:.1f}%",
                "Install date range": f"≤ {cutoff.date()}",
                "LTV30 realized?": "✅ Yes",
                "Use for": "Model evaluation (holdout), Evaluation & Insights page",
            },
            {
                "Dataset": "cfm_pltv_recent",
                "Users": f"{len(recent_df):,}",
                "% of total": f"{len(recent_df)/len(df)*100:.1f}%",
                "Install date range": f"> {cutoff.date()}",
                "LTV30 realized?": "⏳ Not yet",
                "Use for": "Live scoring, Action & Simulation (score recent users)",
            },
        ])
        st.dataframe(preview, use_container_width=True, hide_index=True)

        if len(train_df) < 100:
            st.error("❌ Train set has fewer than 100 users. Reduce the maturity threshold or upload more data.")
            return False
        if len(test_df) < 50:
            st.warning("⚠️ Test set is very small (<50 users). Consider reducing the maturity threshold.")
        if len(recent_df) == 0:
            st.warning("⚠️ No recent users found. All users are mature — Recent dataset will be empty.")

        # ── Custom names ──────────────────────────────────────────
        with st.expander("✏️ Customize dataset names (optional)", expanded=False):
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                name_train = st.text_input("Train name", value="cfm_pltv_train", key="ds_name_train")
            with nc2:
                name_test = st.text_input("Test name", value="cfm_pltv_test", key="ds_name_test")
            with nc3:
                name_recent = st.text_input("Recent name", value="cfm_pltv_recent", key="ds_name_recent")
        # Use defaults if expander not opened
        if "ds_name_train" not in st.session_state:
            name_train, name_test, name_recent = "cfm_pltv_train", "cfm_pltv_test", "cfm_pltv_recent"

        # ── Process ───────────────────────────────────────────────
        st.markdown("---")
        if st.button("🚀 Process & Save Datasets", type="primary", use_container_width=True):
            with st.spinner("Saving datasets…"):
                success = _save_three_datasets(
                    train_df, test_df, recent_df,
                    name_train, name_test, name_recent,
                    source_file=uploaded_file.name,
                    dump_date=dump_date,
                    cutoff=cutoff,
                    maturity_days=int(maturity_days),
                )
            if success:
                st.success("✅ **Three datasets saved and registered!**")
                st.balloons()
                st.markdown(
                    "**Next steps:**  \n"
                    "- Analysis pages default to **cfm_pltv_train**  \n"
                    "- **Evaluation & Insights** and **Action & Simulation** let you switch to **cfm_pltv_test**  \n"
                    "- **Action & Simulation** also lets you score **cfm_pltv_recent** users with the trained model"
                )
                st.markdown("---")
                if st.button("🔄 Regenerate All Pages", type="primary", use_container_width=True,
                             help="Clears all cached data so every page reloads with the new datasets."):
                    regenerate_all_caches()
                    st.success("✅ All caches cleared — navigate to any page to see fresh results.")
                    st.rerun()
                return True
            else:
                st.error("❌ Failed to save. Check errors above.")
                return False

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

    return False


def _save_three_datasets(train_df, test_df, recent_df,
                          name_train, name_test, name_recent,
                          source_file, dump_date, cutoff, maturity_days) -> bool:
    """Save the three split datasets to CSV and register each."""
    from dataset_registry import register_dataset
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        total = len(train_df) + len(test_df) + len(recent_df)

        splits = [
            (name_train,  train_df,  "train",  f"Mature users (install ≤ {cutoff.date()}), 80% split"),
            (name_test,   test_df,   "test",   f"Mature users (install ≤ {cutoff.date()}), 20% holdout"),
            (name_recent, recent_df, "recent", f"Recent users (install > {cutoff.date()}), LTV30 not realized"),
        ]

        for name, chunk, role, split_info in splits:
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            filename = f"{safe_name}.csv"
            filepath = DATA_DIR / filename
            chunk.to_csv(filepath, index=False)

            register_dataset(
                name=name,
                filename=filename,
                rows=len(chunk),
                columns=len(chunk.columns),
                source_file=source_file,
                split_info=split_info,
                extra_meta={
                    "role": role,
                    "dump_date": str(dump_date.date()),
                    "maturity_cutoff": str(cutoff.date()),
                    "maturity_days": maturity_days,
                    "pct_of_total": round(len(chunk) / total * 100, 1) if total > 0 else 0,
                },
            )
            pct = len(chunk) / total * 100 if total > 0 else 0
            st.markdown(f"- ✅ **{name}** → `{filename}` ({len(chunk):,} rows, {pct:.1f}%)")

        # Legacy aliases for backward compatibility
        (DATA_DIR / "cfm_pltv_train.csv").write_bytes((DATA_DIR / f"{name_train.lower().replace(' ','_')}.csv").read_bytes())
        test_path = DATA_DIR / f"{name_test.lower().replace(' ','_')}.csv"
        if test_path.exists():
            (DATA_DIR / "cfm_pltv_test.csv").write_bytes(test_path.read_bytes())

        return True

    except Exception as e:
        st.error(f"❌ Error saving: {str(e)}")
        return False


def regenerate_all_caches():
    """
    Clear all Streamlit cache_data entries and reset session-state dataset
    bindings so every page reloads fresh data on next visit.
    """
    import streamlit as st
    # Clear all @st.cache_data caches (covers _load_csv and any other cached loaders)
    st.cache_data.clear()
    # Drop the in-memory registry mirror so it reloads from disk
    for key in ["_ds_registry", "current_dataset_id", "actual_row_count",
                "model", "model_features", "loaded_model", "loaded_model_metadata"]:
        st.session_state.pop(key, None)


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
