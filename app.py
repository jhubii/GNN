import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Dir-GCN Fraud Detection Demo",
    layout="wide",
)

st.title("Dir-GCN vs Enhanced Dir-GCN – Fraud Detection Results")

# ============================================================
# Mappings
# ============================================================

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


def get_exp_id(config: str) -> str:
    return {
        "C1": "hdim32_L3_drop0p5_lr0p001",
        "C2": "hdim32_L3_drop0p6_lr0p0005",
        "C3": "hdim64_L3_drop0p5_lr0p001",
        "C4": "hdim64_L3_drop0p6_lr0p0005",
    }[config]


# ============================================================
# Helpers
# ============================================================


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_summary(base_dir: Path, conv_type: str):
    """
    summary.json produced by compare_models.py under:
        results/<dataset>/<exp_id>/<conv_type>/summary.json
    """
    path = base_dir / conv_type / "summary.json"
    return load_json(path)


def load_runtime_info(base_dir: Path, conv_type: str):
    """
    Runtime JSON produced by create_diagnostics_plots under:
        results/<dataset>/<exp_id>/runtime/<conv_type>_best_run_inference_runtime.json
    """
    path = base_dir / "runtime" / f"{conv_type}_best_run_inference_runtime.json"
    return load_json(path)


def load_predictions_df(base_dir: Path, conv_type: str):
    """
    Prediction CSV produced by create_diagnostics_plots under:
        results/<dataset>/<exp_id>/predictions/<conv_type>_best_run_predictions.csv
    """
    path = base_dir / "predictions" / f"{conv_type}_best_run_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_problem3_metrics(base_dir: Path, conv_type: str):
    """
    Problem 3 metrics for redundancy / caching:

    CSV:   results/.../problem3_metrics/<conv_type>_best_run_problem3_metrics.csv
    JSON:  results/.../problem3_metrics/<conv_type>_best_run_problem3_summary.json
    """
    csv_path = (
        base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_metrics.csv"
    )
    json_path = (
        base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_summary.json"
    )

    df = pd.read_csv(csv_path) if csv_path.exists() else None
    summary = load_json(json_path)
    return df, summary


def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("Configuration")

dataset_label = st.sidebar.selectbox("Select dataset", list(DATASETS.keys()))
config = st.sidebar.selectbox("Select configuration", CONFIGS)
model_label = st.sidebar.selectbox("Select model", list(MODELS.keys()))

dataset_name = DATASETS[dataset_label]
conv_type = MODELS[model_label]
exp_id = get_exp_id(config)

base_dir = Path("results") / dataset_name / exp_id

# ============================================================
# Current selection info
# ============================================================

st.markdown("### Current Selection")
st.write(f"- **Dataset**: `{dataset_name}`")
st.write(f"- **Configuration**: `{config}` → `{exp_id}`")
st.write(f"- **Model**: `{model_label}` (`{conv_type}`)")
st.write(f"- **Base directory**: `{base_dir}`")

st.markdown("---")

# ============================================================
# SECTION 1 – Problem 1 vs Solution 1 (Classification Performance)
# ============================================================

st.header("Problem 1 vs Solution 1 – Classification Performance")

st.write(
    "This section compares classification performance between the **Baseline "
    "Dir-GCN** and the **Enhanced Dir-GCN (Gated)**. Tables show the mean and "
    "standard deviation across runs. Plots are for the currently selected model."
)

# --- load both summaries to build comparison table ---
baseline_summary = load_summary(base_dir, "dir-gcn")
enh_summary = load_summary(base_dir, "dir-gcn-gated")

if baseline_summary is None or enh_summary is None:
    st.warning(
        "One or both `summary.json` files are missing. "
        "Make sure you have run `compare_models.py` for this dataset & config."
    )
else:
    metrics_for_problem1 = ["val_f1", "test_acc", "test_f1", "test_prec", "test_rec"]

    rows = []
    for m in metrics_for_problem1:
        b_mean, b_std = baseline_summary[m]
        e_mean, e_std = enh_summary[m]
        rows.append(
            {
                "Metric": m,
                "Baseline mean": b_mean,
                "Baseline std": b_std,
                "Enhanced mean": e_mean,
                "Enhanced std": e_std,
            }
        )

    df_perf = pd.DataFrame(rows)
    st.subheader("Classification Metrics (Baseline vs Enhanced)")
    st.dataframe(df_perf, use_container_width=True)
    df_download_button(
        df_perf,
        filename=f"classification_metrics_{dataset_name}_{config}.csv",
        label="Download Classification Metrics (CSV)",
    )

    st.markdown("#### Diagnostic Curves & Confusion Matrix (selected model)")

    cols = st.columns(2)
    # ROC & PR for CURRENT model (conv_type)
    with cols[0]:
        st.markdown("##### ROC Curve")
        roc_path = (
            base_dir / "plots" / conv_type / f"{conv_type}_best_run_roc_curve.png"
        )
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("No ROC curve generated for this model.")

    with cols[1]:
        st.markdown("##### Precision–Recall Curve")
        pr_path = base_dir / "plots" / conv_type / f"{conv_type}_best_run_pr_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)
        else:
            st.info("No Precision–Recall curve generated for this model.")

    cm_path = (
        base_dir / "plots" / conv_type / f"{conv_type}_best_run_confusion_matrix.png"
    )
    st.markdown("##### Confusion Matrix")
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("No confusion matrix generated for this model.")

st.markdown("---")

# ============================================================
# SECTION 2 – Problem 2 vs Solution 2 (Time, Memory, Runtime)
# ============================================================

st.header("Problem 2 vs Solution 2 – Time, Memory, Runtime")

st.write(
    "This section highlights training/testing time, total runtime, memory "
    "consumption, and average inference time per forward pass. "
    "Comparison is again between Baseline and Enhanced models."
)

if baseline_summary is None or enh_summary is None:
    st.warning(
        "Time & memory summaries are unavailable because `summary.json` files "
        "are missing."
    )
else:
    time_mem_metrics = ["train_time", "test_time", "total_time", "mem_mb"]
    rows_tm = []
    for m in time_mem_metrics:
        b_mean, b_std = baseline_summary[m]
        e_mean, e_std = enh_summary[m]
        rows_tm.append(
            {
                "Metric": m,
                "Baseline mean": b_mean,
                "Baseline std": b_std,
                "Enhanced mean": e_mean,
                "Enhanced std": e_std,
            }
        )
    df_time_mem = pd.DataFrame(rows_tm)

    st.subheader("Training / Testing Time and Memory")
    st.dataframe(df_time_mem, use_container_width=True)

    df_download_button(
        df_time_mem,
        filename=f"time_memory_metrics_{dataset_name}_{config}.csv",
        label="Download Time & Memory Metrics (CSV)",
    )

# --- Runtime / inference speedup ---
st.subheader("Inference Runtime (Average Forward Pass)")

runtime_baseline = load_runtime_info(base_dir, "dir-gcn")
runtime_enh = load_runtime_info(base_dir, "dir-gcn-gated")

if runtime_baseline is None or runtime_enh is None:
    st.info(
        "Runtime information not found. Make sure the diagnostics step "
        "(`create_diagnostics_plots`) ran successfully."
    )
else:
    t_base = runtime_baseline.get("avg_inference_time_seconds", None)
    t_enh = runtime_enh.get("avg_inference_time_seconds", None)

    cols_rt = st.columns(3)
    with cols_rt[0]:
        st.metric("Baseline inference time (s)", f"{t_base:.6f}")
    with cols_rt[1]:
        st.metric("Enhanced inference time (s)", f"{t_enh:.6f}")
    with cols_rt[2]:
        # speedup: baseline / enhanced
        speedup = t_base / t_enh if (t_base is not None and t_enh is not None) else 0.0
        st.metric("Enhanced speedup vs Baseline (×)", f"{speedup:.3f}")

st.markdown("---")

# ============================================================
# SECTION 3 – Problem 3 vs Solution 3 (Redundancy, Caching, LCS)
# ============================================================

st.header("Problem 3 vs Solution 3 – Redundancy, Caching, LCS Masking")

st.write(
    "This section focuses on structural redundancy and how often the enhanced "
    "model can **reuse past computations**. Metrics such as cache hit ratio and "
    "aggregation savings come from Problem 3 diagnostics."
)

p3_df_enh, p3_summary_enh = load_problem3_metrics(base_dir, "dir-gcn-gated")

if p3_df_enh is None or p3_summary_enh is None:
    st.info(
        "Problem 3 metrics for the Enhanced model are not available. "
        "Ensure that `collect_problem3_metrics` was executed during training."
    )
else:
    # --- Summary metrics (global) ---
    st.subheader("Global Metrics (Enhanced Dir-GCN)")

    total_naive = p3_summary_enh.get("total_naive_aggregations", 0.0)
    total_actual = p3_summary_enh.get("total_actual_aggregations", 0.0)
    total_saved = p3_summary_enh.get("total_saved_aggregations", 0.0)
    saved_ratio = p3_summary_enh.get("total_saved_ratio", 0.0)
    total_edges = p3_summary_enh.get("total_edges", 0)
    total_rec = p3_summary_enh.get("total_recurring_edges", 0)
    cache_hit_ratio = p3_summary_enh.get("global_cache_hit_ratio", 0.0)

    cols_p3 = st.columns(3)
    with cols_p3[0]:
        st.metric("Total naive aggregations", f"{total_naive:.0f}")
        st.metric("Total actual aggregations", f"{total_actual:.0f}")
    with cols_p3[1]:
        st.metric("Aggregations saved", f"{total_saved:.0f}")
        st.metric("Saved ratio", f"{saved_ratio:.3f}")
    with cols_p3[2]:
        st.metric("Total edges", f"{total_edges}")
        st.metric("Recurring edges", f"{total_rec}")
        st.metric("Cache hit ratio", f"{cache_hit_ratio:.3f}")

    st.subheader("Per-layer Metrics (Enhanced Dir-GCN)")
    st.dataframe(p3_df_enh, use_container_width=True)

    df_download_button(
        p3_df_enh,
        filename=f"problem3_layer_metrics_{dataset_name}_{config}_enhanced.csv",
        label="Download Problem 3 Per-layer Metrics (CSV)",
    )

st.markdown("---")

# ============================================================
# SECTION 4 – Prediction Results (full table + export)
# ============================================================

st.header("Prediction Results – Full Node Table")

st.write(
    "Instead of viewing a single node at a time, all node predictions are shown "
    "in a table. You can scroll, search, and download the full results as CSV "
    "(which can be opened directly in Excel)."
)

pred_df = load_predictions_df(base_dir, conv_type)

if pred_df is None:
    st.info(
        f"No prediction file found at "
        f"`results/{dataset_name}/{exp_id}/predictions/{conv_type}_best_run_predictions.csv`."
    )
else:
    st.subheader(f"Prediction Table – {model_label}")
    st.dataframe(pred_df, use_container_width=True)

    df_download_button(
        pred_df,
        filename=f"predictions_{dataset_name}_{config}_{conv_type}.csv",
        label="Download Prediction Table (CSV)",
    )

    st.caption(
        "Tip: Open the downloaded CSV directly in Excel or any spreadsheet tool "
        "to filter, sort, and further analyze the predictions."
    )
