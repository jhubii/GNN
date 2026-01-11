# Dir-GCN Fraud Detection Demo

This repository contains implementations of:

* **Baseline Dir-GCN**
* **Enhanced Dir-GCN (Gated with LCS Masking & Structural Caching)**

The models are evaluated on **three fraud detection datasets**:

* **Synthetic Fraud** (`fraud-syn`)
* **Online Payments** (`online-payments`)
* **Elliptic Bitcoin Transactions** (`elliptic`)

The project includes utilities for:

* Dataset preprocessing
* Multi-run experiment comparison
* Runtime, memory, and redundancy analysis

---

## 1. Prerequisites

### Recommended Setup

* **OS:** Windows / Linux / macOS
* **Python:** **3.10 (required)**
* **Conda (Miniconda or Anaconda)**

> ⚠️ This project relies on PyTorch Geometric wheels.
> Python versions **above 3.10 are NOT recommended**.

---

## 2. Clone the Repository

```bash
git clone <your-repo-url>.git
cd GNN
```

---

## 3. Environment Setup

Create the Conda environment using the provided configuration:

```bash
conda env create -f environment.yml
conda activate dirgnn
```

Verify installation:

```bash
python -c "import torch; import torch_geometric; print('Environment ready')"
```

---

## 4. Dataset Setup

All datasets must follow the same directory structure.

### 4.1 Create Dataset Directories

From the project root:

```bash
mkdir dataset

mkdir dataset/fraud_syn dataset/elliptic dataset/online_payments
mkdir dataset/fraud_syn/raw dataset/fraud_syn/processed
mkdir dataset/elliptic/raw dataset/elliptic/processed
mkdir dataset/online_payments/raw dataset/online_payments/processed
```

---

### 4.2 Synthetic Fraud (`fraud-syn`)

No download required.
The dataset is generated automatically during preprocessing.

---

### 4.3 Online Payments (`online-payments`)

1. Download the **PaySim / Online Payments Fraud dataset** from Kaggle
2. Place the CSV file inside:

```text
dataset/online_payments/raw/
```

---

### 4.4 Elliptic Bitcoin (`elliptic`)

Download the following files:

* `elliptic_txs_features.csv`
* `elliptic_txs_classes.csv`
* `elliptic_txs_edgelist.csv`

Place all files inside:

```text
dataset/elliptic/raw/
```

---

## 5. Dataset Preprocessing

Run each preprocessing script **once** before training.

```bash
python -m src.prepare_fraud_syn
python -m src.prepare_elliptic
python -m src.prepare_online_payments
```

Processed datasets will be saved to:

```text
dataset/<dataset_name>/processed/
```

---

## 6. Model Training & Evaluation

Run experiments using `compare_models.py`.

> ⚠️ On Windows PowerShell, use **single-line commands**
> (line continuation with `\` is for Bash only).

---

### 6.1 Synthetic Fraud

```bash
python -m src.compare_models --dataset fraud-syn --num_runs 5 --lcs_threshold 0.5 --enable_lcs_masking
```

---

### 6.2 Elliptic

```bash
python -m src.compare_models --dataset elliptic --num_runs 5 --lcs_threshold 0.5 --enable_lcs_masking
```

---

### 6.3 Online Payments

```bash
python -m src.compare_models --dataset online-payments --num_runs 5 --lcs_threshold 0.5 --enable_lcs_masking
```

---

## 7. Streamlit Dashboard (Optional)

The Streamlit dashboard has its **own dependencies**, listed in `requirements.txt`.

### 7.1 Install Streamlit Requirements

Make sure the Conda environment is active:

```bash
conda activate dirgnn
```

Then install the Streamlit-specific requirements:

```bash
pip install -r requirements.txt
```

> This file contains only the packages required for the dashboard (e.g., Streamlit, Pandas, visualization utilities).

---

### 7.2 Run the Streamlit App

From the project root:

```bash
streamlit run app.py
```

Open the printed **local URL** in your browser.

---

### 7.3 Dashboard Notes

* The dashboard reads experiment outputs from the `results/` directory.
* Ensure at least one training run has been completed before opening the app.
* Live inference is supported for the **Online Payments** dataset only.

---

## 8. Results Structure

After training, results are saved under:

```text
results/
├── <dataset>/
│   └── <experiment_id>/
│       ├── dir-gcn/
│       ├── dir-gcn-gated/
│       ├── predictions/
│       ├── runtime/
│       ├── plots/
│       └── problem3_metrics/
```

---

## 9. Configuration Notes

* `--num_runs`
  Number of repeated runs for experimental stability

* `--enable_lcs_masking`
  Enables redundancy-aware LCS masking and structural caching

* `--lcs_threshold`
  Similarity threshold used for LCS-based masking

---

## 10. Common Issues

### PowerShell parsing errors

Use single-line commands or backticks (`) instead of ``.

### Module not found

Ensure the environment is active:

```bash
conda activate dirgnn
```

---

## 11. Quick Start (TL;DR)

```bash
git clone <your-repo-url>.git
cd GNN

conda env create -f environment.yml
conda activate dirgnn

python -m src.prepare_fraud_syn
python -m src.compare_models --dataset fraud-syn --num_runs 5 --lcs_threshold 0.5 --enable_lcs_masking

# Optional dashboard
pip install -r requirements.txt
streamlit run app.py
```

Done 🚀
