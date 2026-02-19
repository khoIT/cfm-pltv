"""
Dataset Registry — Persistent dataset management for the Streamlit app.

Manages multiple datasets that can be independently selected per page.
Persists metadata to JSON so datasets survive app restarts.
"""
import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REGISTRY_FILE = DATA_DIR / "dataset_registry.json"

# Pages that consume datasets (id → display label)
PAGE_IDS = {
    "1_Decision_Definition": "🎯 Decision Definition",
    "2_Features_and_Model": "⚔️ Features & Model",
    "3_Evaluation_and_Insights": "📊 Evaluation & Insights",
    "3b_Late_Payer_Analysis": "🔍 Late Payer Analysis",
    "3c_Temporal_Analysis": "📈 Temporal Analysis",
    "4_Action_and_Simulation": "🎮 Action & Simulation",
    "5_Feedback_and_Learning": "🔄 Feedback & Learning",
    "6_Diagnostics": "🔬 Diagnostics",
}


# ── Registry persistence ───────────────────────────────────────────
def _load_registry() -> dict:
    """Load registry from JSON file."""
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {"datasets": {}, "page_bindings": {}}
    return {"datasets": {}, "page_bindings": {}}


def _save_registry(reg: dict):
    """Save registry to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")


def _sync_to_session(reg: dict):
    """Mirror registry into session state for fast access."""
    st.session_state["_ds_registry"] = reg


def get_registry() -> dict:
    """Get the current registry (from session cache or disk)."""
    if "_ds_registry" in st.session_state:
        return st.session_state["_ds_registry"]
    reg = _load_registry()
    _sync_to_session(reg)
    return reg


# ── Dataset CRUD ───────────────────────────────────────────────────
def register_dataset(name: str, filename: str, rows: int, columns: int,
                     source_file: str = "", split_info: str = "") -> str:
    """
    Register a new dataset.
    Returns the dataset ID.
    """
    reg = _load_registry()
    ds_id = name.lower().replace(" ", "_").replace("-", "_")

    # Ensure unique ID
    base_id = ds_id
    counter = 1
    while ds_id in reg["datasets"]:
        ds_id = f"{base_id}_{counter}"
        counter += 1

    reg["datasets"][ds_id] = {
        "name": name,
        "filename": filename,
        "rows": rows,
        "columns": columns,
        "source_file": source_file,
        "split_info": split_info,
        "created_at": datetime.now().isoformat(),
        "size_mb": round(os.path.getsize(DATA_DIR / filename) / 1e6, 1) if (DATA_DIR / filename).exists() else 0,
    }
    _save_registry(reg)
    _sync_to_session(reg)
    return ds_id


def delete_dataset(ds_id: str):
    """Delete a dataset from the registry (file + metadata)."""
    reg = _load_registry()
    if ds_id not in reg["datasets"]:
        return

    ds = reg["datasets"][ds_id]
    # Remove file
    fpath = DATA_DIR / ds["filename"]
    if fpath.exists():
        try:
            os.remove(fpath)
        except OSError:
            pass

    # Remove from registry
    del reg["datasets"][ds_id]

    # Clear bindings that reference this dataset
    for page_id in list(reg["page_bindings"].keys()):
        if reg["page_bindings"][page_id] == ds_id:
            del reg["page_bindings"][page_id]

    _save_registry(reg)
    _sync_to_session(reg)


def rename_dataset(ds_id: str, new_name: str):
    """Rename a dataset."""
    reg = _load_registry()
    if ds_id in reg["datasets"]:
        reg["datasets"][ds_id]["name"] = new_name
        _save_registry(reg)
        _sync_to_session(reg)


def list_datasets() -> dict:
    """Return dict of {ds_id: metadata}."""
    reg = get_registry()
    # Validate files still exist
    valid = {}
    for ds_id, meta in reg["datasets"].items():
        fpath = DATA_DIR / meta["filename"]
        if fpath.exists():
            valid[ds_id] = meta
    return valid


# ── Page bindings ──────────────────────────────────────────────────
def get_page_dataset(page_id: str) -> str | None:
    """Get the dataset ID bound to a page. Returns None if no binding."""
    reg = get_registry()
    ds_id = reg.get("page_bindings", {}).get(page_id)
    # Validate it still exists
    if ds_id and ds_id in reg.get("datasets", {}):
        fpath = DATA_DIR / reg["datasets"][ds_id]["filename"]
        if fpath.exists():
            return ds_id
    return None


def set_page_dataset(page_id: str, ds_id: str):
    """Bind a dataset to a page."""
    reg = _load_registry()
    if "page_bindings" not in reg:
        reg["page_bindings"] = {}
    reg["page_bindings"][page_id] = ds_id
    _save_registry(reg)
    _sync_to_session(reg)


# ── Data loading ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def _load_csv(filepath: str, max_rows: int = 0, file_mtime: float = 0.0) -> pd.DataFrame:
    """Cached CSV loader with retry for Windows file locks."""
    for attempt in range(5):
        try:
            if max_rows > 0:
                return pd.read_csv(filepath, nrows=max_rows, low_memory=False)
            return pd.read_csv(filepath, low_memory=False)
        except PermissionError:
            if attempt < 4:
                time.sleep(2)
            else:
                raise


def load_dataset(ds_id: str, max_rows: int = 0) -> pd.DataFrame:
    """Load a registered dataset by ID."""
    datasets = list_datasets()
    if ds_id not in datasets:
        raise FileNotFoundError(f"Dataset '{ds_id}' not found in registry.")
    meta = datasets[ds_id]
    fpath = DATA_DIR / meta["filename"]
    mtime = os.path.getmtime(fpath) if fpath.exists() else 0.0
    return _load_csv(str(fpath), max_rows=max_rows, file_mtime=mtime)


def load_page_data(page_id: str, max_rows: int = 0) -> pd.DataFrame | None:
    """Load the dataset bound to a page. Returns None if no binding."""
    ds_id = get_page_dataset(page_id)
    if ds_id is None:
        return None
    return load_dataset(ds_id, max_rows=max_rows)


# ── Sidebar UI ─────────────────────────────────────────────────────
def render_dataset_sidebar(page_id: str):
    """
    Render the Dataset Registry section in the sidebar.
    Shows all datasets, highlights the one bound to the current page,
    and allows the user to switch.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Dataset Registry")

    datasets = list_datasets()

    if not datasets:
        st.sidebar.warning("No datasets registered.")
        st.sidebar.info("📤 Upload data on the **Data Upload** page.")
        st.session_state["data_missing"] = True
        return

    st.session_state["data_missing"] = False

    # Current binding
    current_ds = get_page_dataset(page_id)

    # Build options
    ds_options = {ds_id: f"{meta['name']} ({meta['rows']:,} rows, {meta['size_mb']} MB)"
                  for ds_id, meta in datasets.items()}

    ds_ids = list(ds_options.keys())
    ds_labels = list(ds_options.values())

    # Default index
    if current_ds and current_ds in ds_ids:
        default_idx = ds_ids.index(current_ds)
    else:
        default_idx = 0

    selected_label = st.sidebar.selectbox(
        f"Dataset for this page",
        ds_labels,
        index=default_idx,
        key=f"ds_select_{page_id}",
        help="Each page can use a different dataset. Selection is remembered.",
    )

    selected_idx = ds_labels.index(selected_label)
    selected_ds_id = ds_ids[selected_idx]

    # Update binding if changed
    if selected_ds_id != current_ds:
        set_page_dataset(page_id, selected_ds_id)

    # Show dataset info
    meta = datasets[selected_ds_id]
    st.sidebar.caption(
        f"📄 `{meta['filename']}`  \n"
        f"📊 {meta['rows']:,} rows × {meta['columns']} cols  \n"
        f"💾 {meta['size_mb']} MB"
    )

    # Always load all rows
    st.session_state["max_rows"] = meta["rows"]
    st.session_state["current_dataset_id"] = selected_ds_id

    # Show all datasets summary
    with st.sidebar.expander("All datasets", expanded=False):
        reg_snap = get_registry()
        for did, dmeta in datasets.items():
            marker = "🟢" if did == selected_ds_id else "⚪"
            pages_using = [PAGE_IDS.get(p, p) for p, d in reg_snap.get("page_bindings", {}).items() if d == did]
            pages_str = ", ".join(pages_using) if pages_using else "none"
            st.caption(f"{marker} **{dmeta['name']}** — {dmeta['rows']:,} rows  \n"
                       f"   Used by: {pages_str}")


def check_deleted_dataset_warning(page_id: str) -> bool:
    """
    Check if the page's bound dataset was deleted.
    Returns True if a warning was shown (page should stop).
    """
    reg = get_registry()
    bound_id = reg.get("page_bindings", {}).get(page_id)

    if bound_id is None:
        # No binding — not necessarily an error, just no dataset selected yet
        return False

    if bound_id not in reg.get("datasets", {}):
        st.warning(
            f"⚠️ The dataset previously used by this page (`{bound_id}`) has been deleted.  \n"
            "Please select a new dataset from the sidebar."
        )
        return True

    fpath = DATA_DIR / reg["datasets"][bound_id]["filename"]
    if not fpath.exists():
        st.warning(
            f"⚠️ The dataset file `{reg['datasets'][bound_id]['filename']}` is missing.  \n"
            "Please select a different dataset or re-upload."
        )
        return True

    return False


# ── Migration: auto-register existing files ────────────────────────
def migrate_existing_datasets():
    """
    Auto-register cfm_pltv_train.csv, test1, test2 if they exist
    but aren't in the registry yet.
    """
    reg = _load_registry()
    changed = False

    legacy_files = {
        "cfm_pltv_train": {"filename": "cfm_pltv_train.csv", "name": "Training (default)"},
        "cfm_pltv_test1": {"filename": "cfm_pltv_test1.csv", "name": "Test 1 — OOT Near"},
        "cfm_pltv_test2": {"filename": "cfm_pltv_test2.csv", "name": "Test 2 — OOT Far"},
    }

    for ds_id, info in legacy_files.items():
        fpath = DATA_DIR / info["filename"]
        if fpath.exists() and ds_id not in reg["datasets"]:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    header = fh.readline()
                    ncols = len(header.split(","))
                    nrows = sum(1 for _ in fh)  # remaining lines after header
            except Exception:
                nrows = 0
                ncols = 0
            reg["datasets"][ds_id] = {
                "name": info["name"],
                "filename": info["filename"],
                "rows": nrows,
                "columns": ncols,
                "source_file": "legacy",
                "split_info": "auto-migrated",
                "created_at": datetime.now().isoformat(),
                "size_mb": round(os.path.getsize(fpath) / 1e6, 1),
            }
            changed = True

    if changed:
        _save_registry(reg)
        _sync_to_session(reg)
