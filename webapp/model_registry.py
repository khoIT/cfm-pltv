"""
Model Registry - Persist, load, manage trained models with metadata
"""
import streamlit as st
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil

# Model storage directory
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default model to auto-load on app startup
DEFAULT_MODEL = "pltv_model_20260223_14_16Mrows"


def save_model(model, model_name: str, metadata: dict) -> tuple[bool, str]:
    """
    Save a trained model with metadata to disk.
    
    Args:
        model: Trained model object (e.g., XGBoost, sklearn)
        model_name: Name for the model (will be sanitized)
        metadata: Dict with training info (features, dataset_size, metrics, etc.)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Sanitize model name
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '_', '-')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            return False, "Invalid model name"
        
        # Create model directory
        model_dir = MODELS_DIR / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        model_size_kb = model_path.stat().st_size / 1024
        
        # Add timestamp and save metadata
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_name'] = safe_name
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"Model saved: {safe_name} ({model_size_kb:.1f} KB)"
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg


def load_model(model_name: str):
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        tuple: (model, metadata) or (None, None) if not found
    """
    try:
        model_dir = MODELS_DIR / model_name
        
        if not model_dir.exists():
            return None, None
        
        # Load model
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def list_models() -> list:
    """
    List all saved models with their metadata.
    
    Returns:
        list: List of dicts with model info
    """
    models = []
    
    if not MODELS_DIR.exists():
        return models
    
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get model file size
                    model_path = model_dir / "model.pkl"
                    size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
                    
                    models.append({
                        'name': model_dir.name,
                        'saved_at': metadata.get('saved_at', 'Unknown'),
                        'dataset_size': metadata.get('dataset_size', 0),
                        'n_features': metadata.get('n_features', 0),
                        'model_type': metadata.get('model_type', 'Unknown'),
                        'metrics': metadata.get('metrics', {}),
                        'size_mb': size_mb,
                        'metadata': metadata
                    })
                except Exception as e:
                    st.warning(f"Error reading metadata for {model_dir.name}: {str(e)}")
    
    # Sort by saved_at descending
    models.sort(key=lambda x: x['saved_at'], reverse=True)
    return models


def rename_model(old_name: str, new_name: str) -> bool:
    """
    Rename a saved model.
    
    Args:
        old_name: Current model name
        new_name: New model name
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Sanitize new name
        safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '_', '-')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            st.error("Invalid new model name")
            return False
        
        old_dir = MODELS_DIR / old_name
        new_dir = MODELS_DIR / safe_name
        
        if not old_dir.exists():
            st.error(f"Model '{old_name}' not found")
            return False
        
        if new_dir.exists():
            st.error(f"Model '{safe_name}' already exists")
            return False
        
        # Rename directory
        old_dir.rename(new_dir)
        
        # Update metadata
        metadata_path = new_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['model_name'] = safe_name
            metadata['renamed_at'] = datetime.now().isoformat()
            metadata['previous_name'] = old_name
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
    
    except Exception as e:
        st.error(f"Error renaming model: {str(e)}")
        return False


def delete_model(model_name: str) -> bool:
    """
    Delete a saved model.
    
    Args:
        model_name: Name of the model to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        model_dir = MODELS_DIR / model_name
        
        if not model_dir.exists():
            st.error(f"Model '{model_name}' not found")
            return False
        
        # Delete directory and all contents
        shutil.rmtree(model_dir)
        return True
    
    except Exception as e:
        st.error(f"Error deleting model: {str(e)}")
        return False


def auto_load_default_model():
    """
    Auto-load the default model into session state if no model is loaded yet.
    Called once per session (idempotent).
    """
    if "loaded_model" in st.session_state or "model" in st.session_state:
        return  # already have a model

    model_dir = MODELS_DIR / DEFAULT_MODEL
    if not model_dir.exists():
        return  # default model not on disk

    model, metadata = load_model(DEFAULT_MODEL)
    if model is not None:
        st.session_state["loaded_model"] = model
        st.session_state["loaded_model_name"] = DEFAULT_MODEL
        st.session_state["loaded_model_metadata"] = metadata


def show_model_management_ui():
    """
    Display model management UI in Streamlit.
    Allows users to view, rename, and delete saved models.
    """
    st.markdown("### 🗂️ Saved Models")
    
    models = list_models()
    
    if not models:
        st.info("No saved models found. Train a model to get started!")
        return
    
    st.markdown(f"**Total models:** {len(models)}")
    
    # Display models in expandable sections
    for idx, model_info in enumerate(models):
        with st.expander(f"📦 {model_info['name']}", expanded=(idx == 0)):
            st.markdown(f"**Type:** {model_info['model_type']}")
            st.markdown(f"**Saved:** {model_info['saved_at'][:19].replace('T', ' ')}")
            st.markdown(f"**Dataset Size:** {model_info['dataset_size']:,} rows")
            st.markdown(f"**Features:** {model_info['n_features']}")
            st.markdown(f"**File Size:** {model_info['size_mb']:.2f} MB")
            
            # Display metrics if available
            metrics = model_info.get('metrics', {})
            if metrics:
                st.markdown("**Metrics:**")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        st.markdown(f"  - {metric_name}: {metric_value:.4f}")
                    else:
                        st.markdown(f"  - {metric_name}: {metric_value}")
            
            # Show feature importance chart + top 5 drivers side-by-side
            feat_imp = model_info['metadata'].get('feature_importances', {})
            if feat_imp:
                import plotly.express as px_reg
                sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                top15 = sorted_imp[:15]
                fig_imp = px_reg.bar(
                    x=[v for _, v in top15], y=[k for k, _ in top15], orientation="h",
                    title="Feature Importances (top 15)",
                    labels={"x": "Importance", "y": "Feature"},
                    color=[v for _, v in top15], color_continuous_scale="Tealgrn",
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=350)

                reg_chart_col, reg_drv_col = st.columns([1.4, 1])
                with reg_chart_col:
                    st.plotly_chart(fig_imp, use_container_width=True)
                with reg_drv_col:
                    st.markdown("**🏆 Top 5 pLTV Drivers:**")
                    drv_col1, drv_col2 = st.columns(2)
                    for i, (feat, imp_val) in enumerate(sorted_imp[:5]):
                        with drv_col1 if i % 2 == 0 else drv_col2:
                            st.metric(feat, f"{imp_val:.4f}")
            
            # Action buttons in a single row
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button(f"📥 Load", key=f"load_{model_info['name']}", use_container_width=True):
                    model, metadata = load_model(model_info['name'])
                    if model is not None:
                        st.session_state['loaded_model'] = model
                        st.session_state['loaded_model_name'] = model_info['name']
                        st.session_state['loaded_model_metadata'] = metadata
                        st.success(f"✅ Loaded model: {model_info['name']}")
                        st.rerun()
            with btn_col2:
                if st.button(f"✏️ Rename", key=f"rename_{model_info['name']}", use_container_width=True):
                    st.session_state[f'renaming_{model_info["name"]}'] = True
            with btn_col3:
                if st.button(f"🗑️ Delete", key=f"delete_{model_info['name']}", use_container_width=True):
                    st.session_state[f'confirm_delete_{model_info["name"]}'] = True
            
            # Rename input
            if st.session_state.get(f'renaming_{model_info["name"]}', False):
                new_name = st.text_input(
                    "New name:",
                    value=model_info['name'],
                    key=f"new_name_{model_info['name']}"
                )
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if st.button("✅ Confirm", key=f"confirm_rename_{model_info['name']}"):
                        if rename_model(model_info['name'], new_name):
                            st.success(f"Renamed to: {new_name}")
                            del st.session_state[f'renaming_{model_info["name"]}']
                            st.rerun()
                with col_r2:
                    if st.button("❌ Cancel", key=f"cancel_rename_{model_info['name']}"):
                        del st.session_state[f'renaming_{model_info["name"]}']
                        st.rerun()
            
            # Delete confirmation
            if st.session_state.get(f'confirm_delete_{model_info["name"]}', False):
                st.warning(f"⚠️ Are you sure you want to delete **{model_info['name']}**? This cannot be undone.")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("🗑️ Yes, Delete", key=f"yes_delete_{model_info['name']}", type="primary"):
                        if delete_model(model_info['name']):
                            st.success(f"Deleted: {model_info['name']}")
                            del st.session_state[f'confirm_delete_{model_info["name"]}']
                            # Clear loaded model if it was the deleted one
                            if st.session_state.get('loaded_model_name') == model_info['name']:
                                st.session_state.pop('loaded_model', None)
                                st.session_state.pop('loaded_model_name', None)
                                st.session_state.pop('loaded_model_metadata', None)
                            st.rerun()
                with col_d2:
                    if st.button("❌ Cancel", key=f"cancel_delete_{model_info['name']}"):
                        del st.session_state[f'confirm_delete_{model_info["name"]}']
                        st.rerun()
