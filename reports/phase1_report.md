# Phase 1 — EDA & Preprocessing Report (Exoplanet Regression)

## 1.1 Target analysis & problem definition
- Task: regression
- Target: `pl_rade` (planet radius, Earth radii)
- Dataset shape (raw): 6061 rows × 84 columns
- Missing ratio in target: 0.008

Target summary (non-missing):
- count: 6011
- mean: 5.820
- std: 5.347
- min: 0.310
- 25%: 1.820
- 50%: 2.830
- 75%: 11.954
- max: 77.342

Figures:
- reports/figures/target_pl_rade_hist.png
- reports/figures/target_log1p_pl_rade_hist.png

## 1.2 Feature analysis
Feature table exported to:
- reports/feature_table.csv

## 1.3 Correlations & main dataset insights
Top correlations with target (absolute value):
| feature     |      corr |
|:------------|----------:|
| pl_rade     |  1        |
| pl_radj     |  0.999992 |
| sy_gaiamag  | -0.460559 |
| sy_vmag     | -0.453652 |
| pl_bmasse   |  0.449518 |
| pl_bmassj   |  0.449465 |
| pl_eqt      |  0.436853 |
| pl_radjerr2 | -0.430223 |

Files:
- reports/figures/correlation_heatmap.png
- reports/target_correlations.csv

## 1.4 Preprocessing
Applied:
- Numeric missing values: median imputation
- Categorical missing values: filled with 'missing'
- Target outliers: filtered to [1%, 99%] quantiles

Clean dataset shape: 5890 rows × 87 columns

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
