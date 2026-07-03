# Machine Learning Method Catalog

Use this catalog when `$data-science-modeling` needs a more detailed example for a specific modeling family.

It does not say "use every method." It says: pick the smallest method family that matches the data, objective, metric, and validation design.

## Selection Map

| Problem | Start here | Upgrade when |
|---|---|---|
| data acquisition/scraping | official dataset/API/download | static HTML, then browser automation only if needed |
| binary classification | dummy, logistic regression | nonlinear signal or interactions matter |
| multiclass classification | dummy, multinomial logistic regression | class boundaries are nonlinear |
| regression | mean/median, linear/ridge | residuals show nonlinear pattern |
| ranking/scoring | logistic/ridge, tree ensemble | ordering quality matters more than threshold |
| time series | naive/seasonal naive | baseline misses trend/seasonality/exogenous effects |
| clustering | profile segments first | grouping is useful and stable |
| anomaly detection | rule/robust z-score | anomaly pattern is multivariate |
| dimensionality reduction | PCA | visualization/compression needs nonlinear structure |
| NLP text classification | TF-IDF + linear model | context/semantics matter |
| computer vision | pretrained model/features | image-specific signal matters |
| recommendation | popularity baseline | personalization improves offline metric |
| causal inference | experiment/diff-in-diff/regression | identification assumptions are defensible |

## Data Acquisition and Scraping

Use scraping only when no official dataset, download, or API can answer the question.

Prompt:

```text
Use $data-science-modeling to collect a public dataset for modeling.
Prefer official API/download first. If scraping is needed, save raw data, produce cleaned CSV, validate schema, and create a data card.
```

Baseline path:

1. official CSV/XLSX/download;
2. official API with pagination;
3. static HTML table;
4. embedded JSON;
5. dynamic browser automation;
6. crawling framework only for repeated multi-page collection.

Checks:

- source URL recorded;
- collection timestamp recorded;
- row count and columns validated;
- duplicate rate checked;
- null rates checked;
- raw source saved when reproducibility matters;
- no private/authenticated/paywalled/personal data without permission.

## Classification

### Logistic Regression

Use for binary or multiclass classification when interpretability, speed, and a strong baseline matter.

Prompt:

```text
Use $data-science-modeling to build a logistic regression classifier.
Include majority baseline, preprocessing pipeline, class imbalance handling, F1/ROC-AUC, and coefficient interpretation.
```

Minimal syntax:

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
model.fit(X_train, y_train)

pred = model.predict(X_valid)
proba = model.predict_proba(X_valid)[:, 1]

print("baseline_f1", f1_score(y_valid, baseline.predict(X_valid)))
print("valid_f1", f1_score(y_valid, pred))
print("valid_roc_auc", roc_auc_score(y_valid, proba))
print(classification_report(y_valid, pred))
```

Tune:

```python
{"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}
```

Checks:

- stratified split for imbalanced data;
- no target-derived features;
- metric includes F1/recall/precision if rare positive class.

### k-Nearest Neighbors

Use for small to medium tabular datasets where local similarity matters. Avoid for large/high-dimensional datasets unless reduced first.

Prompt:

```text
Use $data-science-modeling to test KNN after a linear baseline.
Scale numeric features, compare k values, and report runtime.
```

Minimal syntax:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = Pipeline([
    ("preprocess", preprocess),
    ("clf", KNeighborsClassifier()),
])

param_grid = {
    "clf__n_neighbors": [3, 5, 11, 21],
    "clf__weights": ["uniform", "distance"],
}
```

Checks:

- scaling exists for numeric features;
- runtime acceptable;
- validation score beats logistic regression enough to justify slower inference.

### Naive Bayes

Use for text classification or count-like features where a fast baseline is useful.

Prompt:

```text
Use $data-science-modeling to build a Naive Bayes text classifier using TF-IDF/count features.
Compare with logistic regression.
```

Minimal syntax:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_model = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
    ("clf", MultinomialNB()),
])
text_model.fit(X_train_text, y_train)
```

Checks:

- split before vectorization fit;
- inspect top false positives/false negatives;
- compare macro F1 if classes are imbalanced.

### Decision Tree

Use when interpretability and rule extraction matter. Treat as a baseline, not automatically the final model.

Prompt:

```text
Use $data-science-modeling to train a shallow decision tree.
Limit depth, show feature importance, and compare to logistic regression.
```

Minimal syntax:

```python
from sklearn.tree import DecisionTreeClassifier

tree = Pipeline([
    ("preprocess", preprocess),
    ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)),
])
```

Tune:

```python
{
    "clf__max_depth": [3, 4, 6, 8],
    "clf__min_samples_leaf": [10, 20, 50],
}
```

Checks:

- train-validation gap;
- tree depth remains explainable;
- not worse than logistic regression unless interpretability is the goal.

### Random Forest

Use for nonlinear tabular classification when a simple model underfits.

Prompt:

```text
Use $data-science-modeling to compare random forest against logistic regression.
Tune shallowly and report validation F1 plus overfit gap.
```

Minimal syntax:

```python
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
```

Tune:

```python
{
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [None, 8, 16],
    "clf__min_samples_leaf": [1, 5, 10],
}
```

Checks:

- overfit gap;
- feature importance sanity;
- inference/runtime acceptable.

### Gradient Boosting

Use when tabular performance matters and baseline/tree models are not enough.

Prompt:

```text
Use $data-science-modeling to test gradient boosting after logistic regression and random forest.
Use a small search space and compare validation metric, runtime, and overfit gap.
```

Minimal syntax with scikit-learn:

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = Pipeline([
    ("preprocess", preprocess),
    ("clf", HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        random_state=42,
    )),
])
```

If XGBoost/LightGBM/CatBoost is already installed, use it only after baseline comparison.

Checks:

- no blind huge grid;
- validation design matches data;
- improvement justifies complexity.

### Support Vector Machine

Use for medium-sized classification with high-dimensional features. Linear SVM is a strong text baseline.

Prompt:

```text
Use $data-science-modeling to compare linear SVM against logistic regression for text classification.
Report macro F1 and top errors.
```

Minimal syntax:

```python
from sklearn.svm import LinearSVC

svm = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
    ("clf", LinearSVC(class_weight="balanced", random_state=42)),
])
```

Checks:

- probability calibration if probabilities are needed;
- runtime acceptable;
- metric includes macro F1 for imbalance.

## Regression

### Linear Regression

Use as the first real regression baseline.

Prompt:

```text
Use $data-science-modeling to build a regression notebook with median baseline and linear regression.
Metric: MAE. Include residual analysis.
```

Minimal syntax:

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)

lr = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression()),
])
lr.fit(X_train, y_train)

baseline_mae = mean_absolute_error(y_valid, baseline.predict(X_valid))
valid_mae = mean_absolute_error(y_valid, lr.predict(X_valid))
baseline_mae, valid_mae
```

Checks:

- residuals by segment;
- predictions finite;
- MAE/RMSE chosen based on business objective.

### Ridge, Lasso, Elastic Net

Use when linear regression overfits, features are correlated, or feature selection matters.

Prompt:

```text
Use $data-science-modeling to compare ridge, lasso, and elastic net.
Use MAE/RMSE and report coefficients only if stable.
```

Minimal syntax:

```python
from sklearn.linear_model import ElasticNet, Lasso, Ridge

ridge = Pipeline([
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0, random_state=42)),
])
```

Tune:

```python
{"model__alpha": [0.01, 0.1, 1, 10, 100]}
```

Checks:

- scale numeric features;
- compare to plain linear regression;
- inspect coefficient stability if interpreted.

### Tree-Based Regression

Use when residuals show nonlinear patterns.

Prompt:

```text
Use $data-science-modeling to compare random forest regression with linear/ridge baselines.
Metric: MAE. Include residual and overfit checks.
```

Minimal syntax:

```python
from sklearn.ensemble import RandomForestRegressor

rf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )),
])
```

Checks:

- train vs validation MAE;
- segment-level residuals;
- tree model beats ridge enough to justify complexity.

### Gradient Boosting Regression

Use for strong tabular regression after simpler baselines.

Prompt:

```text
Use $data-science-modeling to evaluate gradient boosting regression after ridge and random forest.
Use small tuning and compare MAE.
```

Minimal syntax:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

gbr = Pipeline([
    ("preprocess", preprocess),
    ("model", HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )),
])
```

Checks:

- temporal/group leakage if rows are related;
- overfit gap;
- error distribution, not only mean error.

## Unsupervised Learning

### k-Means Clustering

Use for simple segmentation after feature scaling.

Prompt:

```text
Use $data-science-modeling to create customer segments with k-means.
Start with profiling, scale features, test k=2..8, and describe segment meaning.
```

Minimal syntax:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pipe = Pipeline([
    ("preprocess", preprocess),
    ("cluster", KMeans(n_clusters=4, random_state=42, n_init="auto")),
])
labels = pipe.fit_predict(X)
```

Checks:

- features scaled;
- segments stable across seeds;
- segment profiles are interpretable;
- silhouette is supporting evidence, not the whole decision.

### Hierarchical Clustering

Use when dendrogram/explainable grouping matters and data is not too large.

Prompt:

```text
Use $data-science-modeling to run hierarchical clustering for segment discovery.
Compare linkage options and summarize segment profiles.
```

Minimal syntax:

```python
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = cluster.fit_predict(X_scaled)
```

Checks:

- sample size manageable;
- distance metric matches data;
- segment interpretation is useful.

### DBSCAN

Use for density-based clusters and noise detection.

Prompt:

```text
Use $data-science-modeling to test DBSCAN for irregular clusters and noise.
Scale features and explain sensitivity to eps/min_samples.
```

Minimal syntax:

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_scaled)
```

Checks:

- high-dimensional data may fail;
- report noise ratio;
- do not force clusters if most points become noise.

### Gaussian Mixture Models

Use for soft clustering when probabilistic membership matters.

Prompt:

```text
Use $data-science-modeling to compare Gaussian mixture models for soft segmentation.
Report BIC/AIC and segment profiles.
```

Minimal syntax:

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X_scaled)
probs = gmm.predict_proba(X_scaled)
```

Checks:

- covariance assumptions reasonable;
- components stable;
- probabilities useful for downstream decision.

## Dimensionality Reduction

### PCA

Use for compression, visualization, multicollinearity inspection, or denoising.

Prompt:

```text
Use $data-science-modeling to run PCA for feature compression.
Scale numeric features and report explained variance.
```

Minimal syntax:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

Checks:

- scale features first;
- do not interpret components casually;
- fit PCA only on training data if used in supervised modeling.

### t-SNE and UMAP

Use for visualization, not as proof of real clusters.

Prompt:

```text
Use $data-science-modeling to visualize embeddings with t-SNE/UMAP.
Do not use the plot alone as clustering proof.
```

Minimal syntax:

```python
from sklearn.manifold import TSNE

emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_sample)
```

Checks:

- sample if data is large;
- set random seed;
- do not over-interpret distances.

## Time Series

### Naive and Seasonal Naive

Use as mandatory baseline for forecasting.

Prompt:

```text
Use $data-science-modeling to build a seasonal naive baseline before any forecasting model.
Report MAE by horizon.
```

Minimal syntax:

```python
seasonal_period = 12
valid["forecast"] = train[TARGET].shift(seasonal_period).reindex(valid.index)
```

Checks:

- chronological split;
- no future features;
- horizon-wise metric.

### ARIMA/SARIMA/SARIMAX

Use for univariate or exogenous time series with trend/seasonality.

Prompt:

```text
Use $data-science-modeling to compare seasonal naive with SARIMAX.
Use chronological validation and report MAE/RMSE.
```

Minimal syntax:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)
forecast = fit.forecast(steps=len(y_valid))
```

Checks:

- index frequency valid;
- residual diagnostics;
- forecast horizon error.

### ML Forecasting

Use lag features with tree/linear models when exogenous features matter.

Prompt:

```text
Use $data-science-modeling to build lag-feature ML forecasting.
Create lags only from past data and use time-aware validation.
```

Minimal syntax:

```python
for lag in [1, 2, 3, 12]:
    df[f"lag_{lag}"] = df[TARGET].shift(lag)

df["rolling_3"] = df[TARGET].shift(1).rolling(3).mean()
```

Checks:

- every feature is available at prediction time;
- no centered rolling windows;
- backtest by cutoff date.

## Anomaly Detection

### Robust Z-Score

Use as the first anomaly baseline.

Prompt:

```text
Use $data-science-modeling to detect anomalies with robust z-score baseline.
Then compare Isolation Forest if multivariate patterns matter.
```

Minimal syntax:

```python
median = x.median()
mad = (x - median).abs().median()
robust_z = 0.6745 * (x - median) / mad
anomaly = robust_z.abs() > 3.5
```

Checks:

- threshold is domain-reviewed;
- anomalies are inspected, not just counted.

### Isolation Forest

Use for multivariate anomaly detection.

Prompt:

```text
Use $data-science-modeling to run Isolation Forest for anomaly detection.
Compare to robust z-score baseline and inspect top anomalies.
```

Minimal syntax:

```python
from sklearn.ensemble import IsolationForest

iso = Pipeline([
    ("preprocess", preprocess),
    ("model", IsolationForest(contamination=0.02, random_state=42)),
])
labels = iso.fit_predict(X)
```

Checks:

- contamination rate justified;
- top anomalies manually inspected;
- no supervised metric unless labels exist.

### One-Class SVM

Use for novelty detection on clean "normal" data.

Prompt:

```text
Use $data-science-modeling to test One-Class SVM when normal-only training data exists.
Scale features and compare detected anomaly rate.
```

Minimal syntax:

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
labels = ocsvm.fit_predict(X_normal_scaled)
```

Checks:

- train set mostly normal;
- scaling done;
- runtime acceptable.

## NLP

### TF-IDF + Linear Model

Use as the default text classification baseline.

Prompt:

```text
Use $data-science-modeling to classify text with TF-IDF + logistic regression.
Report macro F1 and inspect top errors.
```

Minimal syntax:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```

Checks:

- split before vectorizer fit;
- remove duplicates across splits;
- macro F1 for imbalanced classes.

### Topic Modeling

Use for exploration, not supervised prediction.

Prompt:

```text
Use $data-science-modeling to run topic modeling on documents.
Start with TF-IDF/NMF or Count/LDA and summarize topics with top terms.
```

Minimal syntax:

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(min_df=5, max_df=0.8, stop_words="english")
X_text = vec.fit_transform(docs)
nmf = NMF(n_components=10, random_state=42)
W = nmf.fit_transform(X_text)
```

Checks:

- topics manually reviewed;
- number of topics not chosen only by aesthetics;
- preprocessing language matches corpus.

### Embeddings + Classifier

Use when semantic similarity matters and embeddings are available.

Prompt:

```text
Use $data-science-modeling to train a classifier on existing text embeddings.
Compare to TF-IDF baseline before using embeddings.
```

Checks:

- embeddings generated without target leakage;
- train/validation documents do not duplicate;
- simple classifier first.

## Computer Vision

### Pretrained Feature Extractor

Use when images are involved and deep learning from scratch is unnecessary.

Prompt:

```text
Use $data-science-modeling to classify images using pretrained features.
Start with a simple classifier on extracted embeddings.
```

Checks:

- train/validation split by entity, not near-duplicate image;
- class imbalance handled;
- augmentations applied only to training.

### Fine-Tuning

Use only if pretrained features underperform and enough labeled data exists.

Prompt:

```text
Use $data-science-modeling to fine-tune a pretrained image classifier.
Include baseline, augmentation, validation curve, and confusion matrix.
```

Checks:

- no duplicate images across splits;
- early stopping;
- validation metric by class.

## Recommendation Systems

### Popularity Baseline

Use before collaborative filtering.

Prompt:

```text
Use $data-science-modeling to build a recommender baseline.
Start with popularity and evaluate precision@k/recall@k.
```

Checks:

- temporal split if recommendations are time-dependent;
- exclude already-seen items in evaluation;
- compare to popularity baseline.

### Collaborative Filtering

Use when user-item interaction history is available.

Prompt:

```text
Use $data-science-modeling to test collaborative filtering against popularity baseline.
Evaluate recall@k and coverage.
```

Checks:

- cold-start users/items documented;
- no future interactions in training;
- metric at k matches product use.

## Causal Inference and Experiments

### A/B Test Analysis

Use when treatment is randomized.

Prompt:

```text
Use $data-science-modeling to analyze an A/B test.
Check randomization balance, estimate lift, confidence interval, and practical significance.
```

Checks:

- unit of randomization;
- sample ratio mismatch;
- multiple testing;
- practical vs statistical significance.

### Difference-in-Differences

Use for before/after treatment with comparison group.

Prompt:

```text
Use $data-science-modeling to run difference-in-differences.
Check parallel trends and report treatment effect with uncertainty.
```

Checks:

- parallel trend evidence;
- treatment timing;
- group/time fixed effects if needed.

### Propensity Score / Matching

Use for observational treatment comparison when assumptions are plausible.

Prompt:

```text
Use $data-science-modeling to estimate treatment effect with propensity scores.
Check covariate balance before and after matching/weighting.
```

Checks:

- overlap/common support;
- balance diagnostics;
- do not imply causality without assumptions.

## Neural Networks

### Multilayer Perceptron

Use only after linear/tree baselines, mostly for nonlinear tabular patterns or embeddings.

Prompt:

```text
Use $data-science-modeling to test an MLP after logistic/ridge and tree baselines.
Scale features, use early stopping, and report overfit gap.
```

Minimal syntax:

```python
from sklearn.neural_network import MLPClassifier

mlp = Pipeline([
    ("preprocess", preprocess),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        early_stopping=True,
        max_iter=300,
        random_state=42,
    )),
])
```

Checks:

- baseline comparison;
- early stopping;
- train-validation gap;
- random seed variability.

### Deep Learning

Use when data type or scale justifies it: images, audio, text, sequence, or large embeddings.

Prompt:

```text
Use $data-science-modeling to design a deep learning notebook only if the baseline underperforms.
Include baseline, data split, training curve, validation metric, and error analysis.
```

Checks:

- enough data;
- leakage-free split;
- validation curve;
- early stopping;
- reproducible config.

## Ensemble and Stacking

### Voting/Averaging

Use when diverse models perform similarly and errors differ.

Prompt:

```text
Use $data-science-modeling to test a simple ensemble after individual models.
Compare ensemble vs best single model.
```

Checks:

- ensemble beats best single model, not just average model;
- added complexity is justified;
- validation only, test set untouched.

### Stacking

Use only after strong single-model baselines.

Prompt:

```text
Use $data-science-modeling to test stacking with out-of-fold predictions.
Avoid leakage and compare to the best individual model.
```

Checks:

- out-of-fold predictions for meta-model;
- no test leakage;
- improvement justifies complexity.

## Model Explainability

### Permutation Importance

Use as a model-agnostic sanity check.

Prompt:

```text
Use $data-science-modeling to add permutation importance for the best model.
Compute on validation data and explain top features carefully.
```

Minimal syntax:

```python
from sklearn.inspection import permutation_importance

imp = permutation_importance(best_model, X_valid, y_valid, scoring=metric, random_state=42)
```

Checks:

- correlated features can distort importance;
- importance is not causality.

### SHAP

Use only if installed or explicitly requested.

Prompt:

```text
Use $data-science-modeling to add SHAP explanation if the package is already installed.
Keep it secondary to validation and error analysis.
```

Checks:

- do not install SHAP casually;
- use sample if data is large;
- explain limitations.

## Final Method Choice Template

Use this final table in notebooks:

| Model | Baseline? | Valid metric | Train metric | Params | Notes |
|---|---:|---:|---:|---|---|
| Dummy | yes | 0.00 | 0.00 | strategy=majority | naive baseline |
| LogisticRegression | yes | 0.74 | 0.76 | C=1 | best simple model |
| RandomForest | no | 0.76 | 0.91 | leaf=5 | overfit risk |

Ship the simplest model that meets the metric target and survives checks.
