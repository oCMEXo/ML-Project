import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.makedirs("reports/figures", exist_ok=True)


df = pd.read_csv("data/raw_exoplanets.csv", comment="#")

target = "pl_rade"

y = df[target].dropna()

plt.figure(figsize=(6, 4))
plt.hist(y, bins=60)
plt.tight_layout()
plt.savefig("reports/figures/target_pl_rade_hist.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(np.log1p(y), bins=60)
plt.tight_layout()
plt.savefig("reports/figures/target_log1p_pl_rade_hist.png", dpi=200)
plt.close()

feature_table = pd.DataFrame({
    "feature": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing_ratio": df.isna().mean(),
    "n_unique": df.nunique(dropna=True)
}).sort_values(["missing_ratio", "n_unique"], ascending=[False, False])

feature_table.to_csv("reports/feature_table.csv", index=False)

numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, center=0)
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png", dpi=200)
plt.close()

target_corr = corr[target].dropna().sort_values(key=np.abs, ascending=False)
target_corr.to_csv("reports/target_correlations.csv")

numeric_cols = df.select_dtypes(include=np.number).columns.drop(target)
categorical_cols = df.select_dtypes(exclude=np.number).columns

df_numeric = df[numeric_cols].copy()
df_categorical = df[categorical_cols].copy()
df_target = df[target]

for col in df_numeric.columns:
    df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

for col in df_categorical.columns:
    df_categorical[col] = df_categorical[col].fillna("missing")

q_low = df_target.quantile(0.01)
q_high = df_target.quantile(0.99)
mask = (df_target >= q_low) & (df_target <= q_high)

df_numeric = df_numeric[mask]
df_categorical = df_categorical[mask]
df_target = df_target[mask]

df_clean = pd.concat([df_numeric, df_categorical, df_target], axis=1)

X_num = df_clean[numeric_cols].copy()
X_scaled = StandardScaler().fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5)
plt.tight_layout()
plt.savefig("reports/figures/pca_2d.png", dpi=200)
plt.close()

df_clean["log_pl_rade"] = np.log1p(df_clean[target])
df_clean["planet_star_radius_ratio"] = df_clean["pl_rade"] / df_clean.get("st_rad", np.nan)
df_clean["log_orbital_period"] = np.log1p(df_clean.get("pl_orbper", np.nan))

df_clean.to_csv("data/clean_exoplanets.csv", index=False)

n_rows, n_cols = df.shape
n_rows_clean = df_clean.shape[0]
missing_target = df[target].isna().mean()
y_desc = y.describe()

top_corr = target_corr.head(8).to_frame("corr").reset_index().rename(columns={"index":"feature"})
top_corr_md = top_corr.to_markdown(index=False)

report = f"""# Phase 1 — EDA & Preprocessing Report (Exoplanet Regression)

## 1.1 Target analysis & problem definition
- Task: regression
- Target: `{target}` (planet radius, Earth radii)
- Dataset shape (raw): {n_rows} rows × {n_cols} columns
- Missing ratio in target: {missing_target:.3f}

Target summary (non-missing):
- count: {int(y_desc['count'])}
- mean: {y_desc['mean']:.3f}
- std: {y_desc['std']:.3f}
- min: {y_desc['min']:.3f}
- 25%: {y_desc['25%']:.3f}
- 50%: {y_desc['50%']:.3f}
- 75%: {y_desc['75%']:.3f}
- max: {y_desc['max']:.3f}

Figures:
- reports/figures/target_pl_rade_hist.png
- reports/figures/target_log1p_pl_rade_hist.png

## 1.2 Feature analysis
Feature table exported to:
- reports/feature_table.csv

## 1.3 Correlations & main dataset insights
Top correlations with target (absolute value):
{top_corr_md}

Files:
- reports/figures/correlation_heatmap.png
- reports/target_correlations.csv

## 1.4 Preprocessing
Applied:
- Numeric missing values: median imputation
- Categorical missing values: filled with 'missing'
- Target outliers: filtered to [1%, 99%] quantiles

Clean dataset shape: {n_rows_clean} rows × {df_clean.shape[1]} columns

Export:
- data/clean_exoplanets.csv

## 1.5 Dimensionality reduction
- PCA (2 components) on scaled numeric features

Figure:
- reports/figures/pca_2d.png

## 1.6 Feature generation
Added features:
- log_pl_rade = log1p(pl_rade)
- planet_star_radius_ratio = pl_rade / st_rad
- log_orbital_period = log1p(pl_orbper)

## 1.7 Next steps (Phase 2)
- Define split strategy and leakage checks (disc_year, discoverymethod, facility)
- Build baseline models (Ridge, RandomForest, GradientBoosting)
- Build full Pipeline with ColumnTransformer and cross-validation
- Evaluate with RMSE/MAE (CV + hold-out test)
- Explainability with SHAP (global + local)
"""

Path("reports/space_report.md").write_text(report, encoding="utf-8")
print("Ready. Put data/raw_exoplanets.csv and run: python space.py")
