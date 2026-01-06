import json
import io
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dir-GCN Fraud Detection Demo", layout="wide")

st.title("Dir-GCN vs Enhanced Dir-GCN – Fraud Detection Results")

# Map labels to internal folder names
DATASETS = {
    "Synthetic Fraud": "fraud-syn",
    "Online Payments": "online-payments",
    "Elliptic Bitcoin": "elliptic",
}

CONFIGS = ["C1", "C2", "C3", "C4"]

MODELS = {
    "Baseline Dir-GCN": "dir-gcn",
    "Enhanced Dir-GCN (Gated)": "dir-gcn-gated",
}


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def render_metrics_table(comparison_json: dict):
    rows = []
    for metric, values in comparison_json.items():
        rows.append(
            {
                "Metric": metric,
                "Baseline (mean)": values["baseline"]["mean"],
                "Baseline (std)": values["baseline"]["std"],
                "Enhanced (mean)": values["enhanced"]["mean"],
                "Enhanced (std)": values["enhanced"]["std"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def get_exp_id(config: str) -> str:
    return {
        "C1": "hdim32_L3_drop0p5_lr0p001",
        "C2": "hdim32_L3_drop0p6_lr0p0005",
        "C3": "hdim64_L3_drop0p5_lr0p001",
        "C4": "hdim64_L3_drop0p6_lr0p0005",
    }[config]


def maybe_show_comparison_bar(base_dir: Path, dataset_name: str, exp_id: str):
    """
    Show the comparison bar graph:
      results/<dataset_name>/<exp_id>/plots/compare_<dataset_name>_<exp_id>_dirgcn_vs_enhanced.png
    """
    compare_path = (
        base_dir / "plots" / f"compare_{dataset_name}_{exp_id}_dirgcn_vs_enhanced.png"
    )
    if compare_path.exists():
        st.subheader("Overall Comparison (Baseline vs Enhanced)")
        st.image(str(compare_path), use_container_width=True)
        st.caption(
            "Figure: Test metrics (Accuracy, F1, Precision, Recall, etc.) for "
            "Baseline Dir-GCN vs Enhanced Dir-GCN."
        )
    else:
        st.info(
            f"No comparison plot found at {compare_path}. "
            "Make sure compare_models.py generated it."
        )


# ====== Sidebar / controls ======
st.sidebar.header("Configuration")

dataset_label = st.sidebar.selectbox("Select dataset", list(DATASETS.keys()))
config = st.sidebar.selectbox("Select configuration", CONFIGS)
model_label = st.sidebar.selectbox("Select model", list(MODELS.keys()))

dataset_name = DATASETS[dataset_label]
conv_type = MODELS[model_label]
exp_id = get_exp_id(config)

base_dir = Path("results") / dataset_name / exp_id

st.markdown("### Current Selection")
st.write(f"- **Dataset**: `{dataset_name}`")
st.write(f"- **Configuration**: `{config}` → `{exp_id}`")
st.write(f"- **Model**: `{model_label}` (`{conv_type}`)")
st.write(f"- **Base directory**: `{base_dir}`")

# ====== Metric table (Problem 1 & 2: performance vs resource usage) ======
comparison_path = base_dir / "comparison_summary.json"
comparison_data = load_json(comparison_path)

if comparison_data is None:
    st.error(f"No comparison_summary.json found at {comparison_path}")
else:
    st.subheader("Problem 1 & 2 – Metric Comparison (Baseline vs Enhanced)")
    render_metrics_table(comparison_data)

    # Comparison bar graph (overall metrics)
    maybe_show_comparison_bar(base_dir, dataset_name, exp_id)

    st.markdown("---")
    st.subheader("Diagnostic Plots")

    cols = st.columns(2)

    # Left column: Baseline Dir-GCN
    with cols[0]:
        st.markdown("#### Baseline Dir-GCN")

        roc_baseline = base_dir / "plots" / "dir-gcn" / "dir-gcn_best_run_roc_curve.png"
        if roc_baseline.exists():
            st.caption("Baseline Dir-GCN ROC Curve")
            st.image(str(roc_baseline))

        pr_baseline = base_dir / "plots" / "dir-gcn" / "dir-gcn_best_run_pr_curve.png"
        if pr_baseline.exists():
            st.caption("Baseline Dir-GCN PR Curve")
            st.image(str(pr_baseline))

        cm_baseline = (
            base_dir / "plots" / "dir-gcn" / "dir-gcn_best_run_confusion_matrix.png"
        )
        if cm_baseline.exists():
            st.caption("Baseline Dir-GCN Confusion Matrix")
            st.image(str(cm_baseline))

    # Right column: Enhanced Dir-GCN (Gated)
    with cols[1]:
        st.markdown("#### Enhanced Dir-GCN (Gated)")

        roc_enh = (
            base_dir
            / "plots"
            / "dir-gcn-gated"
            / "dir-gcn-gated_best_run_roc_curve.png"
        )
        if roc_enh.exists():
            st.caption("Enhanced Dir-GCN ROC Curve")
            st.image(str(roc_enh))

        pr_enh = (
            base_dir / "plots" / "dir-gcn-gated" / "dir-gcn-gated_best_run_pr_curve.png"
        )
        if pr_enh.exists():
            st.caption("Enhanced Dir-GCN PR Curve")
            st.image(str(pr_enh))

        cm_enh = (
            base_dir
            / "plots"
            / "dir-gcn-gated"
            / "dir-gcn-gated_best_run_confusion_matrix.png"
        )
        if cm_enh.exists():
            st.caption("Enhanced Dir-GCN Confusion Matrix")
            st.image(str(cm_enh))

st.markdown("---")

# ====== Problem 3 – Redundancy vs Reuse (Solution 3 metrics) ======
st.header("Problem 3 – Redundant Computation vs Reuse (Solution 3)")

p3_csv_path = (
    base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_metrics.csv"
)
p3_summary_path = (
    base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_summary.json"
)

if p3_csv_path.exists():
    st.subheader("Per-layer Redundancy / Aggregation Metrics")
    p3_df = pd.read_csv(p3_csv_path)
    st.dataframe(p3_df, use_container_width=True)
else:
    st.info(
        f"No Problem 3 metrics CSV found at {p3_csv_path}. "
        "Make sure create_diagnostics_plots generated it."
    )

p3_summary = load_json(p3_summary_path)
if p3_summary is not None:
    st.subheader("Summary (Aggregation Savings & Cache Hit Ratio)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Saved Aggregations Ratio",
            f"{p3_summary.get('total_saved_ratio', 0.0):.4f}",
        )
    with col2:
        st.metric(
            "Global Cache Hit Ratio",
            f"{p3_summary.get('global_cache_hit_ratio', 0.0):.4f}",
        )
    with col3:
        st.metric(
            "Total Recurring Edges",
            f"{p3_summary.get('total_recurring_edges', 0)}",
        )

# ====== Inference Runtime (Scope 2.5) ======
st.markdown("---")
st.header("Inference Runtime (Deployment Efficiency)")

runtime_path = base_dir / "runtime" / f"{conv_type}_best_run_inference_runtime.json"
runtime_data = load_json(runtime_path)

if runtime_data is None:
    st.warning(
        f"No runtime file found at {runtime_path}. "
        "Make sure measure_inference_time exported it."
    )
else:
    avg_t = float(runtime_data.get("avg_inference_time_seconds", 0.0))
    st.write(
        f"- **Selected Model**: `{conv_type}`  \n"
        f"- **Average Inference Time (full-graph forward)**: `{avg_t:.6f}` seconds"
    )

    # If both baseline & enhanced runtimes exist, show speedup (×)
    baseline_rt_path = base_dir / "runtime" / "dir-gcn_best_run_inference_runtime.json"
    enhanced_rt_path = (
        base_dir / "runtime" / "dir-gcn-gated_best_run_inference_runtime.json"
    )

    baseline_rt = load_json(baseline_rt_path)
    enhanced_rt = load_json(enhanced_rt_path)

    if baseline_rt is not None and enhanced_rt is not None:
        b_t = float(baseline_rt.get("avg_inference_time_seconds", 0.0))
        e_t = float(enhanced_rt.get("avg_inference_time_seconds", 0.0))
        if b_t > 0 and e_t > 0:
            speedup = b_t / e_t
            st.write(
                f"- **Baseline Runtime**: `{b_t:.6f}` s  \n"
                f"- **Enhanced Runtime**: `{e_t:.6f}` s  \n"
                f"- **Inference Speedup (Baseline ÷ Enhanced)**: `{speedup:.2f}×`"
            )

# ====== Node-level Prediction (full table + export) ======
st.markdown("---")
st.header("Prediction Results – Full Node Table")

st.write(
    "This table shows the prediction results for **all nodes** in the dataset "
    "for the selected model and configuration. You can also export the results "
    "to CSV or Excel for documentation."
)

# New: use the CSV exported by export_node_predictions(...)
predictions_csv_path = (
    base_dir / "predictions" / f"{conv_type}_best_run_predictions.csv"
)

if not predictions_csv_path.exists():
    st.warning(
        f"No prediction CSV found at {predictions_csv_path}. "
        "Make sure export_node_predictions was called in create_diagnostics_plots."
    )
else:
    df_preds = pd.read_csv(predictions_csv_path)

    st.subheader("Node-level Predictions")
    st.dataframe(df_preds, use_container_width=True)

    # --- Download as CSV ---
    csv_data = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_data,
        file_name=f"{dataset_name}_{conv_type}_predictions.csv",
        mime="text/csv",
    )

    # --- Download as Excel ---
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_preds.to_excel(writer, index=False, sheet_name="Predictions")
    excel_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Predictions as Excel",
        data=excel_buffer,
        file_name=f"{dataset_name}_{conv_type}_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
