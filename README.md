# Dataset Quality Analyzer

A machine learning-powered tool that automatically assesses and predicts CSV dataset quality using a **Random Forest classifier**. It analyzes datasets across multiple quality dimensions and classifies them as **Good** or **Bad** — helping data scientists quickly evaluate whether their data is ML-ready.

---

## Features

- **Automated Quality Assessment** — Analyzes 4 major quality dimensions: missing values, class imbalance, outliers, and skewness/kurtosis
- **Severity Scoring** — Computes a weighted severity score combining all detected issues
- **Feature Engineering** — Generates 9 engineered features (sparsity, aspect ratio, log transforms, interaction terms, composite risk score) to boost model performance
- **Data Leakage Prevention** — Removes binary problem flags before training to avoid target leakage
- **ML-Powered Prediction** — Random Forest classifier with **97.5% test accuracy** and **1.0 ROC-AUC**
- **Cross-Validation** — 5-fold stratified CV with **96.25% ± 6.12%** accuracy

---

## Project Architecture

```
Dataset-Quality-Analyzer/
├── Dataset_Prep.ipynb          # Stage 1 & 2: Metadata generation + quality matrix creation
├── DatasetQualityPrediction.ipynb  # Stage 3: Model training, evaluation & prediction
├── metadata.csv                # Intermediate output — raw quality metadata for 200 datasets
├── quality_matrix.csv          # Final feature matrix (leakage-free) fed to the model
└── README.md
```

---

## Pipeline Overview

The project runs as a **3-stage pipeline**, executed across two Jupyter notebooks:

### Stage 1 — Dataset Analysis & Metadata Generation

> **Notebook:** `Dataset_Prep.ipynb` (Cell 1)

Scans a directory of CSV files and computes raw quality metadata for each dataset.

**Quality checks performed:**

| Check | Condition for Issue |
|---|---|
| Missing Values | Any column has > 5% missing values |
| Class Imbalance | Majority class > 70% of target column |
| Outliers | Values beyond 1.5 × IQR (per numerical feature) |
| Skewness / Kurtosis | \|skew\| > 1 or \|kurtosis\| > 3 |

**Labeling logic:**

A dataset is labeled **Bad** if any of the following are true:
- 3+ quality problems detected
- Severity score ≥ 50
- 2+ problems AND severity ≥ 35

**Output:** `metadata.csv` — 200 datasets with 14 columns (quality label, problem counts, severity scores, dimensional stats)

---

### Stage 2 — Quality Matrix Creation (Leakage Removal + Feature Engineering)

> **Notebook:** `Dataset_Prep.ipynb` (Cell 2)

Transforms the raw metadata into a model-ready feature matrix.

**Removed columns** (data leakage — directly used in labeling):
- `problem_count`, `missing_values_issue`, `class_imbalance_issue`, `outliers_issue`, `skew_kurtosis_issue`, `num_outlier_features`, `num_skewed_features`

**Retained base features (5):**
- `num_rows`, `num_columns`, `severity_score`, `missing_percentage`, `imbalance_ratio`

**Engineered features (9):**

| Feature | Description |
|---|---|
| `dataset_size` | `num_rows × num_columns` |
| `sparsity_score` | `missing_percentage / 100` |
| `aspect_ratio` | `num_rows / (num_columns + 1)` |
| `log_rows` | `log(1 + num_rows)` |
| `log_columns` | `log(1 + num_columns)` |
| `severity_per_1k_rows` | `severity_score / (num_rows/1000 + 1)` |
| `imbalance_severity` | `imbalance_ratio × severity_score` |
| `missing_severity` | `missing_percentage × severity_score / 100` |
| `quality_risk` | Composite: `sparsity×30 + imbalance×25 + severity×45` |

**Output:** `quality_matrix.csv` — 200 rows × 16 columns (14 features + name + label)

---

### Stage 3 — Model Training & Evaluation

> **Notebook:** `DatasetQualityPrediction.ipynb`

Trains a Random Forest classifier on the quality matrix.

**Model configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Estimators | 100 |
| Max Depth | 10 |
| Class Weight | Balanced |
| Test Split | 20% (stratified) |
| Feature Scaling | StandardScaler |

---

## 📈 Results

### Classification Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | 97.50% |
| **ROC-AUC** | 1.0000 |
| **CV Accuracy** | 96.25% ± 6.12% |

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Good | 1.0000 | 0.9583 | 0.9787 | 24 |
| Bad | 0.9412 | 1.0000 | 0.9697 | 16 |

### Top Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | `quality_risk` | 0.3190 |
| 2 | `severity_score` | 0.2979 |
| 3 | `missing_severity` | 0.0757 |
| 4 | `missing_percentage` | 0.0634 |
| 5 | `num_columns` | 0.0539 |

### Dataset Distribution

- **200** total datasets analyzed
- **118** Good (59%) · **82** Bad (41%)

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** / **NumPy** — Data manipulation
- **scikit-learn** — Random Forest, StandardScaler, cross-validation, metrics
- **SciPy** — Skewness & kurtosis calculations
- **Matplotlib** / **Seaborn** — ROC curves, feature importance plots

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### Usage

1. **Prepare metadata** — Place your CSV datasets in the project directory and run the first cell of `Dataset_Prep.ipynb`. This generates `metadata.csv`.

2. **Create quality matrix** — Run the second cell of `Dataset_Prep.ipynb`. This produces `quality_matrix.csv` with engineered features and leakage removed.

3. **Train & evaluate** — Run `DatasetQualityPrediction.ipynb`. The model trains on the quality matrix and outputs accuracy, ROC-AUC, confusion metrics, and feature importance plots.

---

<img width="1786" height="738" alt="Screenshot 2025-10-18 201215" src="https://github.com/user-attachments/assets/e15546f0-40a4-4397-80dc-78eed8f12ec5" />
<img width="1818" height="557" alt="Screenshot 2025-10-18 201243" src="https://github.com/user-attachments/assets/79535729-14c9-4165-b10e-85da04f8fcf5" />
<img width="1711" height="750" alt="Screenshot 2025-10-18 201302" src="https://github.com/user-attachments/assets/1064a6a0-12d1-4fe9-a0c7-f3f883c83428" />


