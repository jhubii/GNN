import json
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

# ====== Metric table ======
comparison_path = base_dir / "comparison_summary.json"
comparison_data = load_json(comparison_path)

if comparison_data is None:
    st.error(f"No comparison_summary.json found at {comparison_path}")
else:
    st.subheader("Metric Comparison (Baseline vs Enhanced)")
    render_metrics_table(comparison_data)

    # 🟥 Comparison bar graph (the one you said is missing)
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

# ====== Node-level Prediction (using precomputed JSON) ======
st.header("Node-level Fraud Prediction")

st.write(
    "Use this panel to inspect the model's prediction for a specific node "
    "(account / address / transaction entity), including its predicted class "
    "and fraud probability."
)

predictions_path = base_dir / "predictions" / f"{conv_type}_node_predictions.json"
predictions = load_json(predictions_path)

if predictions is None:
    st.warning(
        f"No prediction file found at {predictions_path}. "
        "Run export_predictions.py locally and commit the JSON files."
    )
else:
    # Determine valid range
    node_ids = sorted(int(k) for k in predictions.keys())
    min_id, max_id = node_ids[0], node_ids[-1]

    node_id = st.number_input(
        "Node ID",
        min_value=min_id,
        max_value=max_id,
        value=min_id,
        step=1,
        help=f"Valid node indices: {min_id} to {max_id}",
    )

    if st.button("Show Prediction"):
        key = str(node_id)
        if key not in predictions:
            st.error(f"Node ID {node_id} not found in predictions.")
        else:
            rec = predictions[key]
            y_pred = rec.get("y_pred", None)
            y_true = rec.get("y_true", None)
            p0 = rec.get("p0", None)
            p1 = rec.get("p1", None)

            st.subheader("Prediction Result")
            st.write(f"- **Node ID**: `{node_id}`")

            if y_pred is not None:
                st.write(
                    f"- **Predicted Class**: `{y_pred}` (0 = non-fraud, 1 = fraud)"
                )

            if (p0 is not None) and (p1 is not None):
                st.write(
                    f"- **P(non-fraud)**: `{p0:.4f}`  \n- **P(fraud)**: `{p1:.4f}`"
                )

            if y_true is not None:
                if y_true == -1:
                    st.write("- **True Label**: `unlabeled` (no ground-truth provided)")
                else:
                    st.write(f"- **True Label**: `{y_true}` (0 = non-fraud, 1 = fraud)")
