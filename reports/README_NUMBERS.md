# Verified numbers — salary_analysis.ipynb

Every value below is regex-extracted from the notebook's stored cell outputs.


## Dataset

- Total postings: **123,849**
- Postings with any salary: **36,073 (29.1%)**
- salaries.csv rows before filtering: **40,785**
- After currency + $10k–$500k plausibility filter: **40,228** (557 dropped)
- Training rows after inner-joining target to postings: **35,546**
- Train / test: **28,436 / 7,110**
- Features after one-hot encoding: **290**
- Median annual salary (salary_target, n=40,228): **$83,000**
- Target skew: raw **1.56** → log1p **0.06**

## Task 1 — salary regression

Best model: **lightgbm tuned**

| model | CV R² | ± sd | test R² | MAE $ | RMSE $ |
|---|---|---|---|---|---|
| lightgbm tuned | 0.664 | 0.013 | 0.672 | 22,733 | 36,360 |
| stacking | 0.636 | 0.008 | 0.651 | 23,517 | 37,564 |
| lightgbm | 0.642 | 0.013 | 0.649 | 23,918 | 37,526 |
| random forest | 0.633 | 0.014 | 0.641 | 23,621 | 38,227 |
| xgboost | 0.636 | 0.014 | 0.641 | 24,261 | 38,114 |
| catboost | 0.599 | 0.012 | 0.600 | 25,822 | 40,029 |
| ridge | 0.522 | 0.013 | 0.521 | 28,243 | 43,579 |
| linear | 0.522 | 0.013 | 0.521 | 28,250 | 43,572 |
| lasso | 0.506 | 0.011 | 0.502 | 28,815 | 44,308 |
| dummy (median) | -0.000 | 0.000 | -0.000 | 42,297 | 58,895 |

- Best MAE: **$22,733** · RMSE **$36,360**
- Predictions within ±20% of actual: **54.5%**
- R² is computed on **log1p(salary)**; MAE/RMSE are in dollars (expm1 applied to both sides before subtracting).

### Top SHAP features (mean |SHAP|)

| feature | value |
|---|---|
| `exp_level_num` | 0.1526 |
| `log_follower_count` | 0.0660 |
| `desc_char_count` | 0.0499 |
| `title_manager` | 0.0421 |
| `title_engineer` | 0.0401 |
| `exp_missing` | 0.0378 |
| `log_employee_count` | 0.0361 |
| `job_state_CA` | 0.0277 |
| `company_industry_Retail` | 0.0231 |
| `skill_IT` | 0.0227 |

## Task 2 — remote vs on-site

- Rows: **123,849**, positive class **12.3%**
- No-information baseline (always predict on-site): **87.7% accuracy**

| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| random forest | 0.949 | 0.791 | 0.730 |
| lightgbm | 0.948 | 0.796 | 0.709 |
| logistic | 0.916 | 0.672 | 0.587 |

- Tuned threshold: **0.687** (vs 0.5 default)
- Remote class at that threshold: precision **0.731**, recall **0.753**, F1 **0.742**, support **3,049**

## Task 3 — description clusters

- Documents: **24,770** (20% sample) · TF-IDF terms **20,000** · SVD explained variance **20.1%**
- k chosen: **6**
- Silhouette across k=3–10: **0.0414 – 0.0571** (peak at k=8, not k=6)

| k | silhouette | inertia |
|---|---|---|
| 3 | 0.0428 | 4,439 |
| 4 | 0.0536 | 4,321 |
| 5 | 0.0414 | 4,251 |
| 6 | 0.0417 | 4,131 |
| 7 | 0.0461 | 4,063 |
| 8 | 0.0571 | 4,007 |
| 9 | 0.0564 | 3,935 |
| 10 | 0.0568 | 3,869 |

| cluster | label | n | median salary | % with salary | % remote |
|---|---|---|---|---|---|
| 0 | industrial / field service | 9,403 | $58,321 | 30% | 6% |
| 1 | software & engineering | 5,850 | $124,800 | 30% | 21% |
| 2 | 'business owner / earnings' | 120 | — | 0% | 100% |
| 3 | sales, finance, marketing | 4,906 | $93,600 | 34% | 21% |
| 4 | healthcare & nursing | 2,951 | $85,649 | 19% | 3% |
| 5 | retail & store | 1,540 | $39,988 | 16% | 0% |

### Cluster 2 (recruitment spam)

- Count: **120** postings
- Remote: **100%**
- Disclosed salaries: **0%** → **0 postings**, median undefined
- Top terms: `galt, earnings, credit, business, sales, personal, figure, memberships, business owners, saas, promotion, bonus`
