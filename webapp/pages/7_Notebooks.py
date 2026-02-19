"""
Page 7 — Interactive Notebooks
View and interact with Jupyter notebooks directly in the browser.
No JupyterLab installation required.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import render_sidebar, render_top_menu

render_top_menu()
render_sidebar()

st.title("📓 Notebooks")
st.markdown("Explore analysis notebooks directly in your browser — no Jupyter installation needed!")

# Notebook directory
NOTEBOOK_DIR = Path(__file__).resolve().parent.parent.parent / "notebooks"

# Available notebooks
NOTEBOOKS = {
    "1_decision_definition.ipynb": {
        "title": "📋 Decision Definition",
        "description": "Business KPIs, revenue distribution, whale analysis, Gini coefficient, Lorenz curve",
        "icon": "📋"
    },
    "2_feature_exploration.ipynb": {
        "title": "🔍 Feature Exploration", 
        "description": "Feature profiling, correlations, cohort distributions",
        "icon": "🔍"
    },
    "3_modeling_eval.ipynb": {
        "title": "🤖 Modeling & Evaluation",
        "description": "Model training, evaluation metrics, lift curves",
        "icon": "🤖"
    },
    "4_action_simulation.ipynb": {
        "title": "🎯 Action Simulation",
        "description": "Seed selection strategies, ROI simulation, uplift analysis",
        "icon": "🎯"
    },
    "5_feedback_learning.ipynb": {
        "title": "🔄 Feedback & Learning",
        "description": "Time dynamics, stability checks, A/B test planning",
        "icon": "🔄"
    },
}

st.markdown("---")

# Top section: Notebook selector, info, and download/run options
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    with st.expander("📚 Select Notebook", expanded=True):
        selected_nb = st.radio(
            "Choose a notebook:",
            list(NOTEBOOKS.keys()),
            format_func=lambda x: NOTEBOOKS[x]["icon"] + " " + NOTEBOOKS[x]["title"].split(" ", 1)[1],
            label_visibility="collapsed"
        )

with col2:
    nb_info = NOTEBOOKS[selected_nb]
    with st.expander(f"ℹ️ {nb_info['title']}", expanded=True):
        st.markdown(f"**Description:** {nb_info['description']}")

with col3:
    with st.expander("💾 Download & Run", expanded=True):
        nb_path = NOTEBOOK_DIR / selected_nb
        
        if nb_path.exists():
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb_bytes = f.read()
            
            st.download_button(
                label=f"⬇️ Download",
                data=nb_bytes,
                file_name=selected_nb,
                mime="application/x-ipynb+json",
                width='stretch'
            )
            
            st.link_button(
                "🚀 Run in Colab",
                url="https://colab.research.google.com/",
                width='stretch'
            )

# How to use section (collapsible)
with st.expander("💡 How to Use", expanded=False):
    st.markdown("""
**Viewing Notebooks:**
- Select a notebook from the left
- Toggle code/output visibility below
- Expand/collapse cells as needed

**Running Notebooks:**

**Option 1 - Download & Run Locally:**
1. Click "⬇️ Download" button above
2. Install Jupyter: `pip install jupyter`
3. Run: `jupyter notebook`
4. Open the downloaded `.ipynb` file
5. Execute cells with `Shift+Enter`

**Option 2 - Run in Google Colab (Cloud, Free):**
1. Click "⬇️ Download" button above
2. Click "🚀 Run in Colab" or go to [Google Colab](https://colab.research.google.com/)
3. Upload the downloaded notebook
4. Run cells directly in your browser

**Note:** This viewer displays **static** notebook content (last saved outputs). 
To **execute cells and see live results**, use one of the options above.
""")

# Load notebook
nb_path = NOTEBOOK_DIR / selected_nb

if not nb_path.exists():
    st.error(f"❌ Notebook not found: `{selected_nb}`")
    st.info(f"Expected location: `{nb_path}`")
    st.stop()

# Read notebook
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb_content = json.load(f)
    
    # Display notebook metadata in collapsible section
    with st.expander("ℹ️ Notebook Metadata", expanded=False):
        st.markdown(f"**File:** `{selected_nb}`")
        st.markdown(f"**Cells:** {len(nb_content.get('cells', []))}")
        st.markdown(f"**Kernel:** {nb_content.get('metadata', {}).get('kernelspec', {}).get('display_name', 'Unknown')}")

except Exception as e:
    st.error(f"❌ Error loading notebook: {str(e)}")
    st.stop()

# Render notebook cells
st.markdown("### 📄 Notebook Content")

# Add controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    show_code = st.checkbox("Show Code", value=True)
with col_ctrl2:
    show_outputs = st.checkbox("Show Outputs", value=True)
with col_ctrl3:
    expand_all = st.checkbox("Expand All Cells", value=False)

# Render each cell
for idx, cell in enumerate(nb_content.get('cells', [])):
    cell_type = cell.get('cell_type', 'code')
    source = ''.join(cell.get('source', []))
    outputs = cell.get('outputs', [])
    
    # Cell container
    with st.container():
        # Cell header
        if cell_type == 'markdown':
            # Render markdown directly
            st.markdown(source)
        
        elif cell_type == 'code':
            # Code cell with expander
            cell_label = f"Cell {idx + 1}"
            if source.strip():
                first_line = source.split('\n')[0][:50]
                cell_label += f": {first_line}..."
            
            with st.expander(f"💻 {cell_label}", expanded=expand_all):
                if show_code and source.strip():
                    st.code(source, language='python')
                
                if show_outputs and outputs:
                    st.markdown("**Output:**")
                    for output in outputs:
                        output_type = output.get('output_type', '')
                        
                        # Handle different output types
                        if output_type == 'stream':
                            text = ''.join(output.get('text', []))
                            if text.strip():
                                st.text(text)
                        
                        elif output_type == 'execute_result' or output_type == 'display_data':
                            data = output.get('data', {})
                            
                            # HTML output
                            if 'text/html' in data:
                                html_content = ''.join(data['text/html'])
                                components.html(html_content, height=400, scrolling=True)
                            
                            # Plain text
                            elif 'text/plain' in data:
                                text = ''.join(data['text/plain'])
                                st.text(text)
                            
                            # Images
                            elif 'image/png' in data:
                                import base64
                                img_data = data['image/png']
                                st.image(f"data:image/png;base64,{img_data}")
                        
                        elif output_type == 'error':
                            st.error("**Error:**")
                            traceback = '\n'.join(output.get('traceback', []))
                            st.code(traceback, language='python')
        