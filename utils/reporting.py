"""
reporting.py — Helper utilities for rendering Markdown reports with embedded charts.
Used by notebooks and the Streamlit app to generate .md artifacts.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def ensure_dirs():
    """Create reports/ and reports/plots/ if they don't exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename: str, engine: str = "plotly") -> str:
    """Save a figure to reports/plots/ and return the relative markdown image path."""
    ensure_dirs()
    filepath = PLOTS_DIR / filename
    if engine == "plotly":
        fig.write_image(str(filepath))
    elif engine == "matplotlib":
        fig.savefig(str(filepath), bbox_inches="tight", dpi=150)
    elif engine == "echarts":
        # pyecharts render
        fig.render(str(filepath.with_suffix(".html")))
        return f"plots/{filepath.with_suffix('.html').name}"
    return f"plots/{filename}"


def md_image(alt: str, rel_path: str) -> str:
    """Return a Markdown image tag."""
    return f"![{alt}]({rel_path})"


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Build a Markdown table from headers and row data."""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def md_section(title: str, body: str, level: int = 2) -> str:
    """Return a Markdown section with heading."""
    prefix = "#" * level
    return f"{prefix} {title}\n\n{body}\n"


def write_report(filename: str, content: str):
    """Write a Markdown report to the reports/ directory."""
    ensure_dirs()
    filepath = REPORTS_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    print(f"✅ Report written: {filepath}")
    return filepath


def timestamp_line() -> str:
    return f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"


def compile_summary(report_files: List[str], output_filename: str = "CFM_Decision_Intelligence_Summary.md"):
    """Read multiple .md reports and compile them into a single master summary."""
    ensure_dirs()
    parts = []
    parts.append("# CFM Decision Intelligence — Master Summary\n")
    parts.append(timestamp_line())

    # Mapping table
    mapping = [
        ["Decision Definition", "decision_definition.md", "KPIs, ARPU, payer rate"],
        ["Feature / ML Ops Platform", "feature_store_overview.md", "Feature profiling, cohort analysis"],
        ["Models + Evaluation", "model_training.md / evaluation_metrics.md", "Lift, Precision@K, Calibration, ROC"],
        ["Action Simulation", "action_simulation.md", "Top-K ROI, uplift curve"],
        ["Causal Feedback", "feedback_stub.md", "Time dynamics, stability checks"],
    ]
    parts.append("## Artifact → Framework Layer Mapping\n")
    parts.append(md_table(
        ["Framework Layer", "Report Artifact", "Key Metrics"],
        mapping,
    ))
    parts.append("\n---\n")

    for rf in report_files:
        fpath = REPORTS_DIR / rf
        if fpath.exists():
            parts.append(fpath.read_text(encoding="utf-8"))
            parts.append("\n---\n")
        else:
            parts.append(f"> ⚠️ Missing report: `{rf}`\n\n---\n")

    write_report(output_filename, "\n".join(parts))
