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
    """
    Render baseline vs enhanced metrics for the current config.
    Returns the DataFrame so we can allow CSV/Excel download.
    """
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
    return df


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

st.markdown("---")

# ======================================================================
# SECTION 1: Problem 1 vs Solution 1 – Classification Performance
# ======================================================================

st.header("Problem 1 vs Solution 1 – Classification Performance")

comparison_path = base_dir / "comparison_summary.json"
comparison_data = load_json(comparison_path)

if comparison_data is None:
    st.error(f"No comparison_summary.json found at {comparison_path}")
else:
    st.subheader("Metric Comparison (Baseline vs Enhanced)")
    metrics_df = render_metrics_table(comparison_data)

    # Download buttons for metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Metric Comparison (CSV)",
            data=csv_bytes,
            file_name=f"{dataset_name}_{exp_id}_metric_comparison.csv",
            mime="text/csv",
        )
    with col_m2:
        xls_buf = io.BytesIO()
        metrics_df.to_excel(xls_buf, index=False, sheet_name="metrics")
        xls_buf.seek(0)
        st.download_button(
            label="Download Metric Comparison (Excel)",
            data=xls_buf,
            file_name=f"{dataset_name}_{exp_id}_metric_comparison.xlsx",
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        )

    # Overall comparison bar graph
    maybe_show_comparison_bar(base_dir, dataset_name, exp_id)

    st.markdown("#### Diagnostic Plots (Per Model)")
    cols = st.columns(2)

    # Left column: Baseline Dir-GCN
    with cols[0]:
        st.markdown("##### Baseline Dir-GCN")

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
        st.markdown("##### Enhanced Dir-GCN (Gated)")

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

# ======================================================================
# SECTION 2: Problem 2 vs Solution 2 – Time, Memory, Runtime
# ======================================================================

st.header("Problem 2 vs Solution 2 – Time, Memory, Runtime")

# We assume per-model summary.json exists at:
#   base_dir/<conv_type>/summary.json
summary_path = base_dir / conv_type / "summary.json"
summary_data = load_json(summary_path)

if summary_data is None:
    st.warning(f"No summary.json found for this model at {summary_path}")
else:
    # Extract the metrics we care about: train_time, test_time, total_time, mem_mb
    rows = []
    for metric_key in ["train_time", "test_time", "total_time", "mem_mb"]:
        if metric_key in summary_data:
            mean, std = summary_data[metric_key]
            rows.append(
                {
                    "Metric": metric_key,
                    "Mean": mean,
                    "Std": std,
                }
            )
    if rows:
        time_mem_df = pd.DataFrame(rows)
        st.subheader("Training / Testing Time and Memory (selected model)")
        st.dataframe(time_mem_df, use_container_width=True)

        col_tm1, col_tm2 = st.columns(2)
        with col_tm1:
            csv_bytes = time_mem_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Time & Memory Metrics (CSV)",
                data=csv_bytes,
                file_name=f"{dataset_name}_{exp_id}_{conv_type}_time_mem.csv",
                mime="text/csv",
            )
        with col_tm2:
            xls_buf = io.BytesIO()
            time_mem_df.to_excel(xls_buf, index=False, sheet_name="time_mem")
            xls_buf.seek(0)
            st.download_button(
                label="Download Time & Memory Metrics (Excel)",
                data=xls_buf,
                file_name=f"{dataset_name}_{exp_id}_{conv_type}_time_mem.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            )

# Inference runtime (Solution 2.5 / 2.6, also linked to Solution 3)
runtime_path = base_dir / "runtime" / f"{conv_type}_best_run_inference_runtime.json"
runtime_data = load_json(runtime_path)

st.subheader("Inference Runtime (Full-Graph Inference)")

if runtime_data is None:
    st.warning(
        f"No inference runtime file found at {runtime_path}. "
        "Make sure create_diagnostics_plots ran for this config."
    )
else:
    avg_time = runtime_data.get("avg_inference_time_seconds", None)
    if avg_time is not None:
        ms = avg_time * 1000.0
        st.metric(
            label="Average Inference Time per Forward Pass",
            value=f"{avg_time:.4f} s",
            delta=f"{ms:.2f} ms",
        )

st.markdown("---")

# ======================================================================
# SECTION 3: Problem 3 vs Solution 3 – Redundancy & Caching Metrics
# ======================================================================

st.header("Problem 3 vs Solution 3 – Recurring Transactions & Caching")

p3_csv_path = (
    base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_metrics.csv"
)
p3_summary_path = (
    base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_summary.json"
)

if not p3_csv_path.exists():
    st.warning(
        f"No Problem 3 metrics CSV found for this model at {p3_csv_path}. "
        "These are generated by create_diagnostics_plots(...)."
    )
else:
    p3_df = pd.read_csv(p3_csv_path)
    st.subheader("Per-Layer Aggregation & Caching Metrics")
    st.dataframe(p3_df, use_container_width=True)

    col_p3_1, col_p3_2 = st.columns(2)
    with col_p3_1:
        csv_bytes = p3_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Problem 3 Metrics (CSV)",
            data=csv_bytes,
            file_name=f"{dataset_name}_{exp_id}_{conv_type}_problem3_metrics.csv",
            mime="text/csv",
        )
    with col_p3_2:
        xls_buf = io.BytesIO()
        p3_df.to_excel(xls_buf, index=False, sheet_name="problem3")
        xls_buf.seek(0)
        st.download_button(
            label="Download Problem 3 Metrics (Excel)",
            data=xls_buf,
            file_name=f"{dataset_name}_{exp_id}_{conv_type}_problem3_metrics.xlsx",
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        )

    # Quick visual on cache hit ratio or saved_ratio if available
    numeric_cols = []
    for col in ["saved_ratio", "cache_hit_ratio"]:
        if col in p3_df.columns:
            numeric_cols.append(col)
    if numeric_cols:
        st.markdown("##### Problem 3 Key Ratios (per layer)")
        st.bar_chart(p3_df.set_index("layer_idx")[numeric_cols])

# Summary JSON for global cache hit ratio, total saved aggregations, etc.
p3_summary = load_json(p3_summary_path)
if p3_summary is not None:
    st.subheader("Problem 3 Global Summary")
    summary_rows = [{"Metric": k, "Value": v} for k, v in p3_summary.items()]
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

st.markdown("---")

# ======================================================================
# SECTION 4: Prediction Results – Full Table + Export
# ======================================================================

st.header("Prediction Results (All Nodes)")

st.write(
    "Below is the complete prediction table for the selected dataset, configuration, "
    "and model. This includes each node's predicted class and class probabilities. "
    "You can filter directly in the table and export to CSV or Excel."
)

predictions_csv_path = (
    base_dir / "predictions" / f"{conv_type}_best_run_predictions.csv"
)

if not predictions_csv_path.exists():
    st.warning(
        f"No prediction CSV found at {predictions_csv_path}. "
        "These are generated by export_node_predictions(...) in model.py."
    )
else:
    preds_df = pd.read_csv(predictions_csv_path)

    # Optional: simple info
    st.write(f"Total nodes in this prediction table: **{len(preds_df)}**")

    # Show full table
    st.dataframe(preds_df, use_container_width=True, height=500)

    # Download buttons (CSV + Excel)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_bytes,
            file_name=f"{dataset_name}_{exp_id}_{conv_type}_predictions.csv",
            mime="text/csv",
        )
    with col_p2:
        xls_buf = io.BytesIO()
        preds_df.to_excel(xls_buf, index=False, sheet_name="predictions")
        xls_buf.seek(0)
        st.download_button(
            label="Download Predictions (Excel)",
            data=xls_buf,
            file_name=f"{dataset_name}_{exp_id}_{conv_type}_predictions.xlsx",
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        )
