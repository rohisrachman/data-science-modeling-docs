---
name: data-science-modeling
description: Use for data science, data scraping/collection, data analysis, statistical modeling, machine learning, and notebook work where Codex must research comparable methodology, collect or validate data sources, establish a baseline, design structured analysis stages in .ipynb, write efficient step-by-step Python/R/SQL syntax, tune parameters, test and debug code, evaluate results, and report concise accuracy or performance metrics. Trigger for requests involving datasets, APIs, scraping public data, EDA, preprocessing, feature engineering, model selection, baseline models, hyperparameter tuning, cross-validation, ML metrics, notebooks, Google Scholar/GitHub/web research for modeling methods, and production-minded DS code.
---

# Data Science Modeling

## Overview

Build the shortest reliable path from dataset to defensible model result. Combine Ponytail minimalism with senior data scientist, data engineer, ML engineer, QA, and performance habits: simple first, leakage-safe, reproducible, measurable, and tested.

## Operating Rules

- Use the simplest model that answers the question before adding complexity.
- Prefer standard libraries already present in the project: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib/seaborn, requests/BeautifulSoup for simple public scraping, xgboost/lightgbm/catboost only if installed or clearly justified.
- Do not add dependencies for tasks covered by the existing stack.
- Prefer official datasets/downloads/APIs before scraping HTML; use browser automation only when static/API extraction is not enough.
- Do not scrape private, authenticated, paywalled, personal, or restricted data unless the user confirms permission.
- Split train/validation/test before fitted preprocessing to prevent leakage.
- Keep notebooks linear: one purpose per section, runnable top-to-bottom, no hidden state.
- Set random seeds for stochastic steps and show package versions when reproducibility matters.
- Optimize for the user's metric, not a generic "best model".
- Explain assumptions briefly near the code that depends on them.
- Leave one small runnable check for non-trivial transformations or metrics.

## Pipeline

1. Research comparable methodology.
   - If the user asks for current methods, literature, "best", or domain-specific modeling, browse first.
   - Prefer Google Scholar, arXiv, Papers with Code, official library docs, peer-reviewed papers, benchmark reports, and reputable technical blogs.
   - Extract only what affects implementation: problem framing, preprocessing, model families, validation design, metrics, known pitfalls.

2. Establish the baseline project.
   - Define data source, collection method if needed, target, unit of analysis, prediction horizon, leakage boundaries, metric, acceptance threshold, and naive baseline.
   - For classification use dummy/majority, logistic regression, or shallow tree.
   - For regression use mean/median, linear/ridge, or shallow tree.
   - For time series use last value, seasonal naive, moving average, or simple ETS/ARIMA when available.
   - For clustering use no-model profiling first, then k-means/hierarchical/DBSCAN only if the objective needs segmentation.

3. Structure the notebook.
   - Create or revise `.ipynb` sections in this order:
     1. Setup and config
     2. Load data
     3. Optional data collection/scraping
     4. Data audit
     5. Train/validation/test split
     6. EDA on training data
     7. Preprocessing and feature engineering
     8. Baseline model
     9. Candidate models
     10. Parameter tuning
     11. Error analysis
     12. Final test evaluation
     13. Short conclusion
   - Use scripts/modules only when notebook cells become duplicated or hard to test.

4. Write step-by-step syntax.
   - For scraping, validate source legitimacy, save raw data when reproducibility matters, normalize schema, and create a small data card.
   - Use pipelines for fitted transformations where available.
   - Validate schema, missingness, duplicates, target distribution, split integrity, and metric calculation.
   - Keep functions small and named by data task, not by vague phases.
   - For SQL, use CTEs for readable stages and verify row counts after joins.
   - For large data, sample first, then vectorize/chunk/push down to SQL before parallelizing.

5. Apply parameter search conservatively.
   - Start with defaults and a small search space.
   - Use cross-validation only when it matches the data design; use time-aware splits for temporal data and group-aware splits for grouped entities.
   - Prefer `RandomizedSearchCV` or small manual grids before expensive searches.
   - Record best params, validation metric, runtime, and why the chosen model wins.

6. Test code and debug.
   - Run the notebook or the touched cells when feasible.
   - Add assertions for shape, null handling, no target leakage columns, split exclusivity, and metric range.
   - Fix errors from the first failing cell forward; do not mask exceptions unless the user needs a resilient batch job.

7. Test results.
   - Compare against naive baseline and at least one simple interpretable baseline.
   - Check overfit gap: train vs validation/test.
   - Inspect errors by segment, class, date, or business-relevant group.
   - Confirm the model improves the chosen metric enough to justify its complexity.

8. Finish.
   - Show concise results: metric table, best model, best parameters, baseline comparison, and one-sentence interpretation.
   - State the main caveat and the next improvement only if it is evidence-backed.
   - Do not claim "best" unless the search space, metric, and validation design support it.

## Notebook Output Contract

When creating or rewriting an `.ipynb`, produce cells that run in order:

```python
# 1. Setup
import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "target"
METRIC = "roc_auc"  # replace with the project metric
```

```python
# 2. Small data checks
assert TARGET in df.columns
assert df.index.is_unique
assert df[TARGET].notna().all()
```

```python
# 3. Report result briefly
results.sort_values("valid_metric", ascending=False).head()
```

Adapt the exact libraries and metric to the task. Do not force this template when the project already has a better local pattern.

## Metric Selection

- Binary classification: ROC-AUC for ranking, PR-AUC for rare positives, F1/recall/precision when threshold behavior matters, log loss for calibrated probability.
- Multiclass classification: macro F1 for imbalance, accuracy only when classes are balanced, log loss for probability quality.
- Regression: MAE for robust business error, RMSE for large-error penalty, RMSLE for multiplicative error, R2 only as supporting context.
- Time series: MAE/RMSE/MAPE/sMAPE plus backtesting by forecast horizon.
- Clustering: silhouette or Davies-Bouldin only as support; prefer segment stability and business interpretability.

## Final Response

Keep the finish short:

- Files changed or notebook created.
- Data source or scraped file created, if applicable.
- Best model and metric.
- Baseline comparison.
- Best parameters if tuned.
- Tests/checks run.
- One caveat or next step.

Pattern:

`Done: notebook runs through baseline -> tuned model. Best: RandomForest, valid F1 0.81 vs baseline 0.62. Params: max_depth=12, n_estimators=300. Checks: split exclusivity, no null target, metric range. Caveat: test set is small.`
