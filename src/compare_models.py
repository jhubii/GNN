import os
import uuid
import time
import json
from copy import deepcopy

import numpy as np
import psutil

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args  # parsed from CLI


def make_experiment_id(a) -> str:
    """
    Build a short ID from the current hyperparameters so each setting
    gets its own folder.

    Example: hidden_dim=32, num_layers=3, dropout=0.5, lr=0.001
    -> 'hdim32_L3_drop0p5_lr0p001'
    """

    def f(x):
        s = str(x)
        return s.replace(".", "p")

    return f"hdim{a.hidden_dim}_L{a.num_layers}_drop{f(a.dropout)}_lr{f(a.lr)}"


def create_diagnostics_plots(
    conv_type: str,
    label: str,
    best_run_idx: int,
    best_ckpt_path: str,
    data,
    local_args,
    results_root: str,
):
    """
    For the *best* run of a given model, load the best checkpoint and
    generate:

      - Confusion matrix
      - ROC curve           (if binary)
      - Precision–Recall    (if binary)

    Saved under:
        results_root/plots/<conv_type>/
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
    )

    # Rebuild the same data split for this run index
    # same call as during training
    train_mask, val_mask, test_mask = get_dataset_split(
        local_args.dataset, data, local_args.dataset_directory, best_run_idx
    )

    plots_dir = os.path.join(results_root, "plots", conv_type)
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Rebuild the underlying GNN with the same hyper-params
    model_args = deepcopy(local_args)
    model_args.conv_type = conv_type
    model_args.num_features = data.num_features

    # If num_classes is not set on args, infer from labels
    num_classes = int(getattr(model_args, "num_classes", int(data.y.max().item() + 1)))
    model_args.num_classes = num_classes

    model = get_model(model_args)

    # 2) Load checkpoint and extract only the "model." weights
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "", 1)
            cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.long().view(-1).to(device)

    with torch.no_grad():
        logits = model(x, edge_index)  # [N, C]
        probs = torch.softmax(logits, dim=1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = probs.argmax(dim=1)[test_mask].cpu().numpy()

    num_classes = probs.shape[1]

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{label} – Confusion Matrix (best run {best_run_idx + 1})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.tight_layout()
    cm_path = os.path.join(plots_dir, f"{conv_type}_best_run_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")

    # ------------- ROC & PR curves (binary only) -------------
    if num_classes == 2:
        # Assume class 1 is the "fraud"/positive class
        y_score = probs[test_mask, 1].cpu().numpy()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{label} – ROC Curve (best run {best_run_idx + 1})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(plots_dir, f"{conv_type}_best_run_roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close(fig)
        print(f"Saved ROC curve to: {roc_path}")

        # Precision–Recall
        prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label} – Precision–Recall (best run {best_run_idx + 1})")
        fig.tight_layout()
        pr_path = os.path.join(plots_dir, f"{conv_type}_best_run_pr_curve.png")
        plt.savefig(pr_path, dpi=200)
        plt.close(fig)
        print(f"Saved Precision–Recall curve to: {pr_path}")
    else:
        print(f"Skipping ROC/PR curves for {label} (num_classes={num_classes} > 2).")


def run_single_model(base_args, conv_type: str, label: str, results_root: str):
    """
    Train & evaluate a single model type (baseline or enhanced)
    using the SAME hyperparameters from base_args, except conv_type.

    All outputs are saved neatly under:
        results_root/<conv_type>/run_<k>/  (checkpoints, per-run metrics)
        results_root/<conv_type>/summary.json
    """
    # copy args so we don't mutate the original
    local_args = deepcopy(base_args)
    local_args.conv_type = conv_type

    model_results_dir = os.path.join(results_root, conv_type)
    os.makedirs(model_results_dir, exist_ok=True)

    print(f"\n================ {label} (conv_type={conv_type}) ================")

    torch.manual_seed(0)

    # Load dataset once
    dataset, evaluator = get_dataset(
        name=local_args.dataset,
        root_dir=local_args.dataset_directory,
        undirected=local_args.undirected,
        self_loops=local_args.self_loops,
        transpose=local_args.transpose,
    )
    data = dataset._data
    data_loader = DataLoader(
        FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0]
    )

    # Lists to store metrics across runs
    val_scores = []  # best val_f1
    test_accs = []
    test_f1s = []
    test_precs = []
    test_recs = []

    train_times = []
    test_times = []
    total_times = []
    mem_usages_mb = []
    best_ckpt_paths = []  # <--- keep path to best checkpoint per run

    for num_run in range(local_args.num_runs):
        print(
            f"\n---------- {label} | Run {num_run + 1} / {local_args.num_runs} ----------"
        )

        # Same splits for this dataset/run index
        train_mask, val_mask, test_mask = get_dataset_split(
            local_args.dataset, data, local_args.dataset_directory, num_run
        )

        # Build model
        local_args.num_features = data.num_features
        local_args.num_classes = dataset.num_classes

        model = get_model(local_args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=local_args.lr,
            weight_decay=local_args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Folder for this run’s checkpoints & metrics
        run_root = os.path.join(model_results_dir, f"run_{num_run + 1}")
        os.makedirs(run_root, exist_ok=True)

        # Callbacks: early stopping + checkpoint on val_f1
        early_stopping_callback = EarlyStopping(
            monitor="val_f1", mode="max", patience=local_args.patience
        )

        ckpt_dir = os.path.join(run_root, str(uuid.uuid4()))
        os.makedirs(ckpt_dir, exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            dirpath=ckpt_dir,
        )

        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=local_args.num_epochs,
            callbacks=[early_stopping_callback, model_checkpoint_callback],
            profiler="simple" if local_args.profiler else None,
            accelerator="cpu",
            devices=1,
            enable_model_summary=False,  # avoid huge model table logs
        )

        # Training time
        t0_train = time.perf_counter()
        trainer.fit(model=lit_model, train_dataloaders=data_loader)
        t1_train = time.perf_counter()
        train_time = t1_train - t0_train

        # Best validation F1
        best_val_f1 = model_checkpoint_callback.best_model_score.item()
        best_ckpt_path = model_checkpoint_callback.best_model_path
        best_ckpt_paths.append(best_ckpt_path)

        # Test time
        t0_test = time.perf_counter()
        test_results = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]
        t1_test = time.perf_counter()
        test_time = t1_test - t0_test
        total_time = train_time + test_time

        # Memory usage
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = mem_bytes / (1024**2)

        # Extract metrics
        test_acc = float(test_results.get("test_acc", 0.0))
        test_f1 = float(test_results.get("test_f1", 0.0))
        test_prec = float(test_results.get("test_prec", 0.0))
        test_rec = float(test_results.get("test_rec", 0.0))

        # Store
        val_scores.append(best_val_f1)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_precs.append(test_prec)
        test_recs.append(test_rec)
        train_times.append(train_time)
        test_times.append(test_time)
        total_times.append(total_time)
        mem_usages_mb.append(mem_mb)

        # Per-run summary (printed)
        print(f"Best Val F1-score : {best_val_f1:.4f}")
        print(f"Test Accuracy     : {test_acc:.4f}")
        print(f"Test F1-score     : {test_f1:.4f}")
        print(f"Test Precision    : {test_prec:.4f}")
        print(f"Test Recall       : {test_rec:.4f}")
        print(f"Train time (s)    : {train_time:.2f}")
        print(f"Test time  (s)    : {test_time:.2f}")
        print(f"Total time (s)    : {total_time:.2f}")
        print(f"Memory usage (MB) : {mem_mb:.2f}")

        # Save per-run metrics JSON
        run_metrics = {
            "run_index": num_run + 1,
            "best_val_f1": best_val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "train_time": train_time,
            "test_time": test_time,
            "total_time": total_time,
            "mem_mb": mem_mb,
        }
        with open(os.path.join(run_root, "metrics_run.json"), "w") as f:
            json.dump(run_metrics, f, indent=2)

    # Aggregate
    def summarize(values):
        v = np.array(values, dtype=float)
        return v.mean(), v.std()

    results = {
        "label": label,
        "val_f1": summarize(val_scores),
        "test_acc": summarize(test_accs),
        "test_f1": summarize(test_f1s),
        "test_prec": summarize(test_precs),
        "test_rec": summarize(test_recs),
        "train_time": summarize(train_times),
        "test_time": summarize(test_times),
        "total_time": summarize(total_times),
        "mem_mb": summarize(mem_usages_mb),
        "num_runs": int(local_args.num_runs),
    }

    # Print final summary
    print(f"\n===== {label} FINAL SUMMARY over {local_args.num_runs} runs =====")
    for key in [
        "val_f1",
        "test_acc",
        "test_f1",
        "test_prec",
        "test_rec",
        "train_time",
        "test_time",
        "total_time",
        "mem_mb",
    ]:
        mean, std = results[key]
        print(f"{key:15s}: {mean:.4f} ± {std:.4f}")

    # Save summary JSON for this model
    summary_path = os.path.join(model_results_dir, "summary.json")
    serializable = {
        k: (float(v[0]), float(v[1])) if isinstance(v, tuple) else v
        for k, v in results.items()
        if k not in ("label",)
    }
    serializable["label"] = label
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {label} summary to: {summary_path}")

    # ---------------- Diagnostics plots for best run ----------------
    best_run_idx = int(np.argmax(np.array(val_scores)))
    best_ckpt_path = best_ckpt_paths[best_run_idx]

    print(
        f"\n[Diagnostics] {label}: using best run index {best_run_idx + 1} "
        f"with best_val_f1={val_scores[best_run_idx]:.4f}"
    )

    create_diagnostics_plots(
        conv_type=conv_type,
        label=label,
        best_run_idx=best_run_idx,
        best_ckpt_path=best_ckpt_path,
        data=data,
        local_args=local_args,
        results_root=results_root,
    )

    return results


def plot_comparison(
    baseline_results,
    enhanced_results,
    dataset_name: str,
    results_root: str,
    exp_id: str,
):
    """Simple bar chart comparing Accuracy, F1, Precision, Recall."""
    import matplotlib.pyplot as plt

    metrics = ["test_acc", "test_f1", "test_prec", "test_rec"]
    labels = ["Accuracy", "F1-score", "Precision", "Recall"]

    baseline_vals = [baseline_results[m][0] for m in metrics]  # means
    enhanced_vals = [enhanced_results[m][0] for m in metrics]  # means

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_vals, width, label=baseline_results["label"])
    ax.bar(x + width / 2, enhanced_vals, width, label=enhanced_results["label"])

    ax.set_ylabel("Score")
    ax.set_title(f"Dir-GCN vs Enhanced Dir-GCN ({dataset_name}, {exp_id})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()

    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(
        plots_dir, f"compare_{dataset_name}_{exp_id}_dirgcn_vs_enhanced.png"
    )
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved comparison plot to: {out_path}")


if __name__ == "__main__":
    # Build experiment ID from hyperparameters so each setting is separate
    exp_id = make_experiment_id(args)

    # Root folder for everything related to this dataset + hyperparams
    results_root = os.path.join("results", args.dataset, exp_id)
    os.makedirs(results_root, exist_ok=True)

    # Save hyperparameters for this experiment
    hyperparams = {
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "num_runs": args.num_runs,
        "alpha": args.alpha,
        "learn_alpha": args.learn_alpha,
        "undirected": args.undirected,
        "self_loops": args.self_loops,
        "transpose": args.transpose,
        "jk": args.jk,
        "normalize": args.normalize,
    }
    with open(os.path.join(results_root, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Experiment ID: {exp_id}")
    print(f"Hyperparameters saved to: {os.path.join(results_root, 'hyperparams.json')}")

    # Baseline Dir-GCN
    baseline_results = run_single_model(
        base_args=args,
        conv_type="dir-gcn",
        label="Baseline Dir-GCN",
        results_root=results_root,
    )

    # Enhanced Dir-GCN (gated)
    enhanced_results = run_single_model(
        base_args=args,
        conv_type="dir-gcn-gated",
        label="Enhanced Dir-GCN (Gated)",
        results_root=results_root,
    )

    # Print a compact side-by-side summary
    print("\n================ OVERALL COMPARISON ================")
    comparison_summary = {}
    for name, key in [
        ("Val F1 (best)", "val_f1"),
        ("Test Acc      ", "test_acc"),
        ("Test F1       ", "test_f1"),
        ("Test Prec     ", "test_prec"),
        ("Test Rec      ", "test_rec"),
        ("Train time (s)", "train_time"),
        ("Total time (s)", "total_time"),
        ("Mem usage (MB)", "mem_mb"),
    ]:
        b_mean, b_std = baseline_results[key]
        e_mean, e_std = enhanced_results[key]
        print(
            f"{name}: "
            f"Baseline = {b_mean:.4f} ± {b_std:.4f} | "
            f"Enhanced = {e_mean:.4f} ± {e_std:.4f}"
        )
        comparison_summary[key] = {
            "baseline": {"mean": float(b_mean), "std": float(b_std)},
            "enhanced": {"mean": float(e_mean), "std": float(e_std)},
        }

    # Save overall comparison JSON
    comparison_path = os.path.join(results_root, "comparison_summary.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nSaved overall comparison summary to: {comparison_path}")

    # Plot bar chart for this hyperparameter setting
    plot_comparison(
        baseline_results,
        enhanced_results,
        dataset_name=args.dataset,
        results_root=results_root,
        exp_id=exp_id,
    )
