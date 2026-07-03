# Baseline Research: Machine Learning Methods

This is the baseline research material for `data-science-modeling`.

Use it before writing a notebook when the task needs method selection, literature grounding, or a defensible baseline. The goal is not to try every method. The goal is to choose the smallest valid method family, compare it to a baseline, and upgrade only when the evidence supports it.

Research date: 2026-07-02

## Core Sources

Primary and official sources used:

- scikit-learn supervised learning: https://scikit-learn.org/stable/supervised_learning.html
- scikit-learn unsupervised learning: https://scikit-learn.org/stable/unsupervised_learning.html
- scikit-learn metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn pipelines/composite estimators: https://scikit-learn.org/stable/modules/compose.html
- scikit-learn feature extraction/text: https://scikit-learn.org/stable/modules/feature_extraction.html
- scikit-learn common pitfalls and data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- statsmodels state space/SARIMAX: https://www.statsmodels.org/stable/statespace.html
- XGBoost docs: https://xgboost.readthedocs.io/en/stable/
- LightGBM docs: https://lightgbm.readthedocs.io/en/stable/
- CatBoost docs: https://catboost.ai/docs/en/
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/index
- GitHub syntax and latest research tracker: ./github-and-latest-research.md
- Data scraping workflow and examples: ../examples/data-scraping.md

## Non-Negotiable Baseline Rules

1. Split before fitted preprocessing.
2. Use a naive baseline before candidate models.
3. Use a simple interpretable baseline before complex models.
4. Choose validation by data structure, not convenience.
5. Choose metric by decision objective.
6. Tune only after a model beats baseline.
7. Keep search space small until evidence justifies expansion.
8. Report train vs validation/test when overfit is plausible.
9. Do not claim "best" without defining search space and validation design.

scikit-learn's common pitfalls guidance is especially important: preprocessing, feature selection, imputation, scaling, PCA, and other transformations must be learned from training data only. Pipelines help prevent inconsistent preprocessing and leakage.

## Method Selection Overview

| Family | Use first when | Upgrade when | Baseline |
|---|---|---|---|
| Dummy models | every supervised task | never final unless no signal | majority, stratified, mean, median |
| Linear/logistic | tabular baseline, interpretability | nonlinear residuals/interactions remain | linear/ridge/logistic |
| Regularized linear | many/correlated features | linear underfits | ridge/lasso/elastic net |
| KNN | local similarity matters, small data | scaling and runtime acceptable | scaled KNN vs linear |
| Naive Bayes | sparse counts/text | linear model underperforms | multinomial/complement NB |
| SVM/LinearSVC | medium/high-dimensional data | probability or scale needs differ | linear SVM after logistic |
| Decision tree | explainable rules | tree overfits or underfits | shallow tree |
| Random forest | nonlinear tabular | boosting clearly wins | RF after simple tree |
| Gradient boosting | strong tabular performance | baseline/tree models insufficient | HGB/XGB/LGBM/CatBoost |
| Neural net/MLP | embeddings/nonlinear data | tabular baselines underperform | MLP after linear/tree |
| Clustering | segmentation/exploration | clusters stable and useful | profiling, k-means |
| Dimensionality reduction | compression/visualization | nonlinear structure needed | PCA |
| Anomaly detection | rare unusual points | multivariate pattern needed | robust z-score/rules |
| Time series | temporal forecast | naive misses trend/seasonality | naive/seasonal naive |
| NLP | text features | TF-IDF baseline underperforms | TF-IDF + linear |
| CV/deep learning | images/audio/video | pretrained features insufficient | pretrained features |
| Recommender | user-item ranking | personalization improves metric | popularity |
| Causal | treatment effect | assumptions defensible | randomized experiment/regression |

## Supervised Classification

### Dummy Classifier

Use as the first baseline for every classification task.

Baseline choices:

- majority class for class imbalance visibility;
- stratified dummy for random-label comparison;
- prior-probability dummy for probability metrics.

Metric:

- accuracy only if classes are balanced and equal-cost;
- macro F1 for multiclass imbalance;
- PR-AUC/F1/recall/precision for rare positives;
- ROC-AUC for ranking when both classes are reasonably represented.

Red flags:

- model accuracy barely beats majority class;
- high accuracy with low minority recall;
- no confusion matrix or per-class report.

### Logistic Regression

Use as the first real classifier for tabular and sparse text features.

Good for:

- fast baseline;
- interpretable directionality;
- linear decision boundary;
- probability output.

Initial parameters:

- `max_iter=1000`;
- `class_weight="balanced"` for imbalance;
- tune `C` over `[0.01, 0.1, 1, 10]`.

Upgrade when:

- residual/error analysis shows nonlinear interactions;
- underfit persists after feature engineering;
- model cannot separate classes enough for target metric.

### KNN

Use only when local similarity is plausible and data is not too large.

Requirements:

- numeric scaling;
- sensible distance metric;
- small/medium sample size.

Initial parameters:

- `n_neighbors`: 3, 5, 11, 21;
- `weights`: uniform, distance.

Red flags:

- high-dimensional sparse data;
- slow inference;
- no scaling.

### Naive Bayes

Use for text/count features and very fast baselines.

Variants:

- GaussianNB for continuous roughly Gaussian features;
- MultinomialNB for counts/TF-IDF;
- ComplementNB for imbalanced text;
- BernoulliNB for binary indicators.

Upgrade when:

- feature dependence hurts performance;
- linear model clearly outperforms.

### SVM / LinearSVC

Use for high-dimensional sparse data, especially text classification.

Default path:

1. TF-IDF + LogisticRegression
2. TF-IDF + LinearSVC
3. Calibrate only if probabilities are required

Red flags:

- RBF SVM on large data without runtime check;
- no scaling for numeric features;
- using SVM probability output casually.

### Decision Tree

Use for interpretable rule baseline.

Initial controls:

- `max_depth`: 3, 4, 6, 8;
- `min_samples_leaf`: 10, 20, 50;
- cost-complexity pruning if needed.

Red flags:

- train score far above validation;
- deep tree presented as interpretable;
- no comparison with logistic regression.

### Random Forest

Use for nonlinear tabular classification when simple models underfit.

Initial controls:

- `n_estimators`: 200-500;
- `min_samples_leaf`: 1, 5, 10;
- `max_depth`: None, 8, 16;
- `class_weight="balanced"` for imbalance.

Red flags:

- feature importance interpreted causally;
- overfit gap ignored;
- runtime/inference not considered.

### Gradient Boosting

Use for strong tabular performance after simple baselines.

Options:

- scikit-learn `HistGradientBoostingClassifier` for built-in baseline;
- XGBoost when installed and justified;
- LightGBM for large-scale efficient GBDT;
- CatBoost when categorical features are important and package is available.

Initial tuning:

- learning rate: 0.03, 0.05, 0.1;
- max iterations/trees: 100-500;
- max leaves/depth: small first;
- regularization/min child samples for overfit control.

Red flags:

- huge grid before baseline;
- no early stopping/validation;
- using test set for tuning.

## Supervised Regression

### Dummy Regressor

Use mean or median baseline.

Metric:

- MAE for business-readable error;
- RMSE for penalizing large errors;
- RMSLE for multiplicative/skewed targets;
- R2 only as supporting context.

### Linear Regression

Use as the first real regression baseline.

Checks:

- residual plots;
- segment-level error;
- finite predictions;
- target transformation if heavily skewed.

### Ridge, Lasso, Elastic Net

Use when features are many, correlated, or noisy.

Initial tuning:

- alpha: 0.01, 0.1, 1, 10, 100;
- l1 ratio for ElasticNet: 0.1, 0.5, 0.9.

Requirements:

- scale numeric features;
- fit preprocessing inside pipeline.

### Tree/Forest/Boosting Regression

Use after linear baselines when residuals are nonlinear.

Default upgrade path:

1. DummyRegressor
2. Linear/Ridge
3. RandomForestRegressor
4. HistGradientBoostingRegressor or installed boosting library

Red flags:

- train MAE far below validation MAE;
- no residual analysis;
- model only evaluated on average error.

## Unsupervised Learning

### Clustering

Start with profiling before modeling. Clustering is useful only if segments are stable and interpretable.

Methods:

- k-means for compact spherical-ish clusters;
- hierarchical clustering for dendrogram/interpretable grouping;
- DBSCAN for density-based clusters and noise;
- Gaussian mixture for soft membership.

Metrics:

- silhouette, Davies-Bouldin, Calinski-Harabasz as support;
- stability across seeds/samples;
- segment interpretability;
- downstream usefulness.

Red flags:

- picking cluster count from silhouette alone;
- forcing clusters where there is no useful segmentation;
- no feature scaling.

### Dimensionality Reduction

Use PCA first.

Methods:

- PCA for linear compression and denoising;
- t-SNE/UMAP for visualization only;
- NMF for non-negative parts/topic-like decomposition.

Rules:

- scale before PCA when units differ;
- fit PCA only on train if used in supervised pipeline;
- do not treat t-SNE/UMAP plot as proof of clusters.

## Time Series Forecasting

Start with temporal baselines.

Baselines:

- last value;
- seasonal naive;
- moving average;
- simple exponential smoothing where appropriate.

Models:

- SARIMAX/state-space via statsmodels for trend/seasonality/exogenous features;
- lag-feature ML with linear/tree/boosting models;
- deep sequence models only when data scale and objective justify them.

Validation:

- chronological split;
- rolling-origin backtest;
- horizon-wise metrics.

Metrics:

- MAE/RMSE;
- MAPE/sMAPE only when denominator behavior is valid;
- error by forecast horizon.

Red flags:

- random split;
- centered rolling windows;
- future exogenous features;
- feature engineering after looking at validation/test horizon.

## NLP

Start with TF-IDF + linear model.

Baseline path:

1. majority baseline;
2. TF-IDF + LogisticRegression;
3. TF-IDF + LinearSVC;
4. embeddings/transformers only when semantic errors dominate or requested.

Tasks:

- classification/sentiment;
- topic modeling;
- document clustering;
- keyword extraction;
- semantic similarity;
- text regression;
- multilingual text.

Metrics:

- macro F1 for imbalanced classification;
- PR-AUC/F1/recall for rare class;
- topic coherence/manual review for topic models;
- retrieval precision@k/recall@k for similarity/retrieval.

Red flags:

- vectorizer fit before split;
- duplicate text across train/validation;
- English stopwords blindly used on multilingual data;
- transformer fine-tuning without TF-IDF baseline.

## Computer Vision, Audio, Video, Multimodal

Use pretrained features before training from scratch.

Baseline path:

1. simple heuristic or majority baseline;
2. pretrained embedding/features + linear classifier;
3. fine-tuning pretrained model;
4. train from scratch only with enough labeled data and compute.

Validation:

- split by entity/source to avoid near-duplicate leakage;
- class-level metrics;
- inspect errors visually/audibly.

Red flags:

- duplicate/near-duplicate media across splits;
- augmentation applied to validation/test;
- no per-class confusion matrix.

## Recommendation Systems

Start with popularity.

Baseline path:

1. popularity;
2. item-item/user-user similarity;
3. matrix factorization/collaborative filtering;
4. hybrid features;
5. learning-to-rank if labels and infrastructure justify it.

Validation:

- temporal split for event data;
- leave-one-out or user-level split when appropriate;
- exclude already-seen items in evaluation.

Metrics:

- precision@k;
- recall@k;
- MAP@k/NDCG@k;
- coverage/diversity as supporting metrics.

Red flags:

- random interaction split causing future leakage;
- recommending items already consumed;
- optimizing RMSE when product needs top-k ranking.

## Anomaly Detection

Start with a robust rule baseline.

Baseline path:

1. domain rule or robust z-score/MAD;
2. IsolationForest for multivariate unsupervised anomalies;
3. OneClassSVM for novelty detection with clean normal data;
4. supervised classifier if labels exist.

Metrics:

- precision/recall if labels exist;
- alert volume;
- analyst review hit rate;
- top-k inspection.

Red flags:

- arbitrary contamination rate;
- no manual inspection;
- treating unsupervised anomaly scores as ground truth.

## Causal Inference and Experiments

Do not treat predictive modeling as causal inference.

Baseline path:

1. randomized A/B test if possible;
2. regression adjustment;
3. difference-in-differences when pre/post treated/control groups exist;
4. matching/propensity weighting with balance diagnostics;
5. instrumental variables only when instrument assumptions are defensible.

Checks:

- unit of treatment;
- sample ratio mismatch for experiments;
- covariate balance;
- common support;
- parallel trends for DiD;
- sensitivity analysis.

Red flags:

- causal language from observational prediction;
- no identification assumption;
- no uncertainty interval.

## Explainability

Use explanation as a diagnostic, not proof.

Methods:

- coefficients for linear models;
- impurity/permutation importance for tree models;
- partial dependence/ICE for feature-response inspection;
- SHAP only if installed/needed;
- error analysis by segment is often more useful than feature attribution.

Rules:

- importance is not causality;
- correlated features distort importance;
- explain validation behavior, not only training behavior.

## Model Selection Template

Use this in notebooks:

| Candidate | Purpose | Valid metric | Train metric | Complexity | Decision |
|---|---|---:|---:|---|---|
| Dummy | naive baseline | 0.00 | 0.00 | very low | reference |
| Linear/logistic | simple baseline | 0.74 | 0.76 | low | keep |
| Random forest | nonlinear candidate | 0.76 | 0.91 | medium | overfit risk |
| Gradient boosting | performance candidate | 0.79 | 0.84 | medium/high | choose if worth it |

Choose the simplest candidate that:

- beats baseline materially;
- uses valid validation;
- survives leakage checks;
- has acceptable runtime;
- can be explained enough for the use case.

## Research-To-Notebook Checklist

Before coding:

- define task type;
- define data source and collection method if data must be scraped;
- define target and unit of analysis;
- identify leakage boundaries;
- pick baseline;
- pick metric;
- pick validation design;
- decide candidate model family;
- define minimal tuning space;
- define expected checks.

Before finishing:

- show baseline comparison;
- show best model and params;
- show train-validation/test metric;
- run leakage checks;
- run metric sanity checks;
- state caveat;
- avoid unsupported "best model" claims.
