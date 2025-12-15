# 🌌 Explainable Regression for Exoplanet Radius Prediction

## 📌 Project Overview

This repository contains an individual course project for **Classical Machine Learning**. The goal is to demonstrate a complete ML pipeline — from exploratory data analysis (EDA) and feature engineering to preprocessing and preparation for modelling — using real-world astronomical data.

The focus is on predicting the **radius of exoplanets** (target: `pl_rade`) from observational, orbital, and stellar features, with particular emphasis on:

- data quality analysis
- transparent preprocessing decisions
- explainability and interpretability of results
- reproducibility of experiments

---

## 🎯 Machine Learning Task

- **Type:** Supervised Learning
- **Task:** Regression
- **Target variable:** `pl_rade` — planet radius (in Earth radii)

The radius is a continuous physical quantity that helps characterize planetary types (e.g. rocky vs gaseous), so regression is an appropriate and well-justified approach.

---

## 📊 Dataset

- **Source:** NASA Exoplanet Archive — Planetary Systems Composite Data (Confirmed Planets)
- **License:** NASA Open Data (suitable for educational and research use)
- **Size:** ~6,000 observations, 80+ features
- **Types:** numerical and categorical

Key groups of variables:

- planetary parameters (radius, orbital period, semi-major axis)
- stellar properties (mass, radius, effective temperature)
- discovery metadata (method, year, facility)

---

## 🧭 Project Structure

```
ML-Project/
├── phase1.py
├── README.md
├── data/
│   ├── raw_exoplanets.csv
│   └── clean_exoplanets.csv
└── reports/
	├── phase1_report.md
	└── figures/
		├── target_pl_rade_hist.png
		├── target_log1p_pl_rade_hist.png
		├── correlation_heatmap.png
		└── pca_2d.png
```

---

## 🔬 Phase 1 — EDA & Preprocessing (Completed)

> Phase 1 contains a reproducible EDA, preprocessing and feature engineering pipeline with an accompanying report (`reports/phase1_report.md`). Highlights:

### 1.1 Target Analysis & Problem Definition

- Distribution analysis of `pl_rade`
- Motivation for regression and log-transform exploration

### 1.2 Feature Analysis

- Feature meanings and data types
- Missing value ratios and cardinality analysis

### 1.3 Correlations & Insights

- Correlation matrix for numerical features
- Top features correlated with the target and initial physical hypotheses

### 1.4 Preprocessing

- Median imputation for numerical features
- Explicit handling of missing categorical values
- Outlier filtering based on quantiles

### 1.5 Dimensionality Reduction

- PCA on scaled numerical features and visualizations

### 1.6 Feature Generation

- Log-transformed target
- Planet-to-star radius ratio
- Log-transformed orbital period

### 1.7 EDA Report

- Auto-generated report saved at `reports/phase1_report.md`

---

## ▶️ How to Run the Project

### 1️⃣ Requirements

Python **3.10+** is recommended. Install the main dependencies with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```

### 2️⃣ Download the Dataset

Download the Planetary Systems Composite Data from the NASA Exoplanet Archive:

https://exoplanetarchive.ipac.caltech.edu

Save the CSV as:

```
data/raw_exoplanets.csv
```

> ⚠️ Important: Do not remove lines starting with `#` in the downloaded file — the ingestion script (`phase1.py`) handles them automatically.

### 3️⃣ Run Phase 1

From the project root directory run:

```bash
python phase1.py
```

### 4️⃣ Output

After successful execution the following artifacts will be produced:

- Clean dataset: `data/clean_exoplanets.csv`
- EDA report: `reports/phase1_report.md`
- Figures: `reports/figures/*.png`

---

If you want a hand expanding the project (modelling, cross-validation, model explainability plots), tell me what you'd like to prioritize next — I'd be glad to help! ✅
