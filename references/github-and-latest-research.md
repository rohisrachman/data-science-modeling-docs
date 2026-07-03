# GitHub Syntax References and Latest Research Tracker

Research date: 2026-07-02

Use this file as a routing map when `$data-science-modeling` needs implementation syntax or current research direction. Do not copy code blindly. Use repositories for syntax patterns, then adapt to the project's split, metric, leakage boundary, and existing dependencies.

## Rule

GitHub examples are syntax references, not validation proof. Recent papers are research leads, not permission to skip baseline.

Default order:

1. local project pattern;
2. official library docs;
3. official GitHub examples;
4. reputable research repo;
5. paper implementation if it is maintained and needed.

## GitHub Syntax References

| Area | Repository | Use for | Notes |
|---|---|---|---|
| General ML | https://github.com/scikit-learn/scikit-learn/tree/main/examples | classification, regression, clustering, pipelines, model selection, inspection, text | Prefer for scikit-learn-native syntax and leakage-safe patterns. |
| XGBoost | https://github.com/dmlc/xgboost/tree/master/demo | boosted trees, multiclass, ranking, Dask/distributed examples | Use only after simple/tree baselines. |
| LightGBM | https://github.com/microsoft/LightGBM/tree/master/examples | efficient GBDT, ranking, Python/R examples | Good for large tabular data; tune conservatively. |
| CatBoost | https://github.com/catboost/tutorials | categorical-heavy tabular data, notebooks | Useful when categorical features dominate and package exists. |
| statsmodels | https://github.com/statsmodels/statsmodels/tree/main/examples | statistical models, time series, SARIMAX/state-space examples | Use for econometrics/statistical modeling and interpretable time-series baselines. |
| Hugging Face notebooks | https://github.com/huggingface/notebooks | transformer/NLP/CV/audio notebooks | Use only after classical baseline unless user explicitly requests transformers. |
| Sentence Transformers | https://github.com/huggingface/sentence-transformers/tree/main/examples | semantic similarity, retrieval, embeddings, cross-encoders | Compare against TF-IDF cosine or TF-IDF linear baseline. |
| PyOD | https://github.com/pyod-team/pyod | anomaly detection models | Start with robust z-score/domain rules first. |
| Scrapy | https://github.com/scrapy/scrapy | larger crawling/scraping projects | Use only when simple requests/API download is not enough. |
| Playwright Python | https://github.com/microsoft/playwright-python | dynamic page/browser automation scraping | Use only when static/API extraction fails. |
| TabPFN | https://github.com/PriorLabs/TabPFN | tabular foundation model examples | Treat as advanced candidate; check license, data size, hardware, and baseline first. |
| Chronos | https://github.com/amazon-science/chronos-forecasting | pretrained time-series forecasting | Compare against naive/seasonal naive/SARIMAX. |
| TimesFM | https://github.com/google-research/timesfm | Google time-series foundation model | Use for research/prototyping after chronological baselines. |
| Uni2TS/Moirai | https://github.com/SalesforceAIResearch/uni2ts | universal time-series transformers | Use for foundation-model comparison, not default forecasting. |

## Syntax Search Workflow

When a user asks for syntax:

1. Identify the task family: tabular, NLP, time series, anomaly, recommender, causal, CV.
2. Search the local project first with `rg`.
3. If no local pattern exists, use the official repo examples above.
4. Copy only the minimal pattern: imports, estimator, fit/predict/evaluate.
5. Wrap fitted preprocessing in a pipeline where possible.
6. Add leakage checks that the example repo may not include.
7. Add the project metric and baseline comparison.

For scraping syntax, prefer API/download/static HTML examples before Scrapy or browser automation. Keep raw data, validate schema, and document collection time.

## Latest Research Directions To Track

### Tabular Foundation Models

Track:

- TabPFN and later TabPFN model reports;
- lightweight/educational TabPFN reimplementations;
- tabular foundation models vs GBDT comparisons;
- limitations on row count, feature count, licensing, and hardware.

Representative sources:

- https://github.com/PriorLabs/TabPFN
- https://arxiv.org/abs/2511.03634

Baseline rule:

Use TabPFN-style models only after comparing against:

1. dummy baseline;
2. logistic/ridge/linear baseline;
3. random forest or gradient boosting baseline.

Checks:

- license permits intended use;
- dataset size fits model limits;
- no unsupported preprocessing;
- GPU/CPU runtime acceptable;
- validation split is still project-specific.

### Time-Series Foundation Models

Track:

- Chronos;
- TimesFM;
- Moirai/Uni2TS;
- TimeFound and other time-series foundation models;
- finance/economics-specific evaluations.

Representative sources:

- https://github.com/amazon-science/chronos-forecasting
- https://github.com/google-research/timesfm
- https://github.com/SalesforceAIResearch/uni2ts
- https://arxiv.org/abs/2503.04118
- https://arxiv.org/abs/2507.00945
- https://arxiv.org/abs/2605.21504

Baseline rule:

Do not use a time-series foundation model before:

1. last-value baseline;
2. seasonal naive;
3. moving average or ETS/SARIMAX where appropriate;
4. chronological/rolling validation.

Checks:

- no random split;
- horizon-wise metrics;
- exogenous variables available at forecast time;
- compare zero-shot/foundation model against simple temporal baselines.

### NLP and Embedding Models

Track:

- Hugging Face Transformers;
- Sentence Transformers;
- SetFit/PEFT style efficient fine-tuning;
- retrieval and cross-encoder reranking examples.

Representative sources:

- https://github.com/huggingface/notebooks
- https://github.com/huggingface/sentence-transformers/tree/main/examples
- https://huggingface.co/docs/transformers/index

Baseline rule:

Do not fine-tune transformers before:

1. majority baseline;
2. TF-IDF + LogisticRegression;
3. TF-IDF + LinearSVC;
4. duplicate-leakage check.

Checks:

- exact or near-duplicate text across splits;
- macro F1 for imbalanced classes;
- language-specific tokenization/stopwords;
- runtime and hardware constraints.

### Gradient Boosting Research and Practice

Track:

- XGBoost;
- LightGBM;
- CatBoost;
- categorical feature handling;
- monotonic/interaction constraints;
- ranking objectives.

Representative sources:

- https://github.com/dmlc/xgboost/tree/master/demo
- https://github.com/microsoft/LightGBM/tree/master/examples
- https://github.com/catboost/tutorials
- https://xgboost.readthedocs.io/en/stable/
- https://lightgbm.readthedocs.io/en/stable/
- https://catboost.ai/docs/en/

Baseline rule:

Use boosted trees after a simple baseline and a random forest or shallow tree baseline.

Checks:

- small search space first;
- early stopping if available;
- train-valid gap;
- categorical handling is correct;
- test set untouched.

### Anomaly Detection

Track:

- PyOD models;
- Isolation Forest;
- One-Class SVM;
- deep anomaly detection only if labels/data scale justify it.

Representative sources:

- https://github.com/pyod-team/pyod
- https://scikit-learn.org/stable/modules/outlier_detection.html

Baseline rule:

Start with domain thresholds or robust z-score/MAD.

Checks:

- contamination rate justified;
- top anomalies inspected;
- alert volume acceptable;
- labels used only if truly available.

## Research Refresh Checklist

When the user asks for "latest" methods:

1. Search official docs/release notes for library updates.
2. Search GitHub repo examples for maintained syntax.
3. Search arXiv/Hugging Face Papers/Papers with Code for current papers.
4. Prefer papers with public code and reproducible benchmarks.
5. Record publication date and code URL.
6. Mark unvalidated methods as research candidates, not production defaults.

## What To Add To A Notebook From Research

Only add:

- method family;
- why it fits the task;
- preprocessing constraints;
- validation design;
- metric;
- minimal hyperparameters;
- known limitations;
- source links in a markdown cell.

Do not add:

- long literature summaries;
- untested paper claims;
- heavyweight dependencies without approval;
- a new model family without a baseline comparison.
