# Data Science Modeling

He does not start with XGBoost.
He starts with the baseline. Then he proves the upgrade is worth it.

`data-science-modeling` is a Codex skill for collecting/validating data, building structured data science notebooks, statistical models, and machine learning workflows that are simple first, leakage-safe, reproducible, tested, and measured.

![status](https://img.shields.io/badge/status-local_skill-blue)
![agent](https://img.shields.io/badge/agent-Codex-black)
![notebook](https://img.shields.io/badge/output-ipynb-orange)
![license](https://img.shields.io/badge/license-MIT-green)

Source first · Baseline first · Notebook top-to-bottom · No leakage · Small tuning · Clear metric · Short finish

---

## Why This Exists

Most data science agents overbuild:

- They jump to advanced models before a baseline exists.
- They scrape brittle pages before checking official downloads or APIs.
- They tune parameters before the split is trustworthy.
- They report accuracy even when the classes are imbalanced.
- They fit preprocessing before splitting the data.
- They produce notebooks that only run if you execute cells in the exact accidental order.

This skill forces a better path:

1. research methods only when it changes implementation;
2. collect data from the least fragile source;
3. define the project baseline;
4. structure the notebook;
5. write step-by-step syntax;
6. tune conservatively;
7. test code;
8. test results;
9. report the metric briefly.

The goal is not fancy modeling. The goal is a defensible result.

## Before / After

You ask:

```text
Buat model prediksi churn dari dataset ini.
```

Without this skill, the agent may:

- install extra libraries;
- run EDA on the whole dataset;
- fit encoders before splitting;
- train random forest, gradient boosting, and neural network immediately;
- report `accuracy = 0.94`;
- skip baseline and leakage checks.

With `data-science-modeling`:

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
TARGET = "churn"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

assert set(X_train.index).isdisjoint(set(X_valid.index))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_f1 = f1_score(y_valid, baseline.predict(X_valid))

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)
valid_f1 = f1_score(y_valid, model.predict(X_valid))

baseline_f1, valid_f1
```

It starts boring. Boring survives review.

## Numbers

This skill has not been benchmarked publicly yet. Do not invent performance claims.

Use this measurement plan if you want GitHub-ready numbers:

| Metric | What to measure |
|---|---|
| notebook cells | fewer redundant cells vs no-skill baseline |
| leakage checks | count of explicit split/preprocessing/target checks |
| reproducibility | random seed, version note, top-to-bottom execution |
| model quality | validation/test metric vs baseline |
| runtime | notebook execution time |
| complexity | dependencies added, files added, custom abstractions |

Recommended benchmark setup:

1. Pick 8-12 realistic data science tasks.
2. Run the same agent with and without the skill.
3. Use the same dataset, prompt, and environment.
4. Score only runnable artifacts: `.ipynb`, scripts, and final metrics.
5. Count failures: leakage, non-runnable notebook, missing baseline, wrong metric, untested code.

Report numbers only after that.

## How It Works

Before writing code, the agent stops at the first rung that holds:

```text
1. Does this model need to exist?
2. Does a naive baseline answer enough?
3. Is a standard model enough?
4. Does the existing stack solve it?
5. Can the notebook stay linear?
6. Can tuning stay small?
7. Only then: add complexity.
```

The ladder runs after understanding the dataset and objective. It is lazy about unnecessary complexity, never lazy about data validation.

Never simplify away:

- train/validation/test split discipline;
- leakage checks;
- data source and scraping/data-quality checks;
- metric correctness;
- target validation;
- reproducibility;
- error handling that prevents wrong results;
- clear caveats.

## Install

The skill already exists locally at:

```text
/Users/macos/.codex/skills/data-science-modeling/SKILL.md
```

Codex discovers skills from:

```text
~/.codex/skills/
```

So the installed path is correct.

To install manually on another machine:

```bash
mkdir -p ~/.codex/skills/data-science-modeling
cp SKILL.md ~/.codex/skills/data-science-modeling/SKILL.md
```

Then start a new Codex session and invoke:

```text
$data-science-modeling
```

## Usage

Use the skill explicitly:

```text
Use $data-science-modeling to build a structured notebook for churn prediction.
Target: churn. Metric: F1. Include baseline, tuning, tests, and concise results.
```

Use it for data collection:

```text
Use $data-science-modeling to collect public data for this analysis.
Prefer official API/download first. If scraping is needed, save raw data, cleaned CSV, data card, and quality checks.
```

Use it for research-driven modeling:

```text
Use $data-science-modeling.
Research comparable methods for monthly inflation forecasting, then build a leakage-safe notebook.
Do not use random split.
```

Use it for debugging:

```text
Use $data-science-modeling to review this notebook.
Check leakage, split design, preprocessing, metric choice, and suspiciously high validation score.
```

Use it for improvement:

```text
Use $data-science-modeling to improve this model.
Start from baseline comparison, run error analysis, then tune only if the evidence supports it.
```

## Commands

This is a Codex skill, not a plugin with slash commands. Use prompt-level commands:

| Command | What it does |
|---|---|
| `$data-science-modeling` | Activate the skill for the current task |
| `Use $data-science-modeling to create notebook...` | Create a structured DS notebook |
| `Use $data-science-modeling to debug notebook...` | Review and fix modeling issues |
| `Use $data-science-modeling to research methodology...` | Search comparable methods before implementation |
| `Use $data-science-modeling to tune model...` | Tune conservatively after baseline |

## Pipeline

### 1. Research Comparable Methodology

Browse when the user asks for:

- latest/current methods;
- best model;
- methodology;
- paper-based approach;
- domain-specific modeling;
- benchmarking;
- Google Scholar or web research.

Prefer:

- Google Scholar;
- arXiv;
- Papers with Code;
- official library documentation;
- peer-reviewed papers;
- benchmark reports;
- reputable technical blogs.

Extract only implementation-relevant information:

- problem framing;
- preprocessing;
- model family;
- validation design;
- metrics;
- known pitfalls;
- leakage risks.

### 2. Establish Baseline Project

Define:

- target variable;
- unit of analysis;
- prediction horizon;
- leakage boundaries;
- metric;
- acceptance threshold;
- naive baseline;
- simple interpretable baseline.

Baseline choices:

| Task | Baseline |
|---|---|
| binary classification | majority class, logistic regression, shallow tree |
| multiclass classification | majority class, multinomial logistic regression |
| regression | mean/median, linear regression, ridge |
| time series | last value, seasonal naive, moving average |
| clustering | profiling first, then k-means/hierarchical/DBSCAN if needed |

### 3. Structure The Notebook

Notebook sections:

1. Setup and config
2. Load data
3. Data audit
4. Train/validation/test split
5. EDA on training data
6. Preprocessing and feature engineering
7. Baseline model
8. Candidate models
9. Parameter tuning
10. Error analysis
11. Final test evaluation
12. Short conclusion

Rules:

- The notebook must run top-to-bottom.
- Do not rely on hidden state.
- Do not use validation/test data to design features.
- Move code into scripts only when duplication becomes a real problem.

### 4. Write Step-By-Step Syntax

Required checks:

```python
assert TARGET in df.columns
assert df.index.is_unique
assert df[TARGET].notna().all()
```

Split before fitted preprocessing:

```python
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Use pipelines:

```python
model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
```

For SQL:

- use CTEs;
- check row counts after joins;
- avoid many-to-many explosions;
- keep feature windows before the prediction timestamp.

### 5. Tune Parameters Conservatively

Start with defaults.

Then use a small search:

```python
param_grid = {
    "clf__C": [0.1, 1, 10],
    "clf__class_weight": [None, "balanced"],
}
```

Use `RandomizedSearchCV` or a small `GridSearchCV`.

Match validation to data structure:

| Data structure | Split |
|---|---|
| IID tabular | stratified/random split |
| time series | time-aware split |
| users/accounts/products repeated | group-aware split |
| rare target | stratified split |

### 6. Test Code And Debug

Minimal checks:

```python
assert len(X_train) == len(y_train)
assert len(X_valid) == len(y_valid)
assert set(X_train.index).isdisjoint(set(X_valid.index))
assert np.isfinite(valid_score)
```

Add context checks:

- no leakage columns;
- no null target;
- expected row count after joins;
- metric in valid range;
- prediction shape matches target shape;
- encoder handles unknown categories.

Debug from the first failing cell forward. Do not hide failures with broad `try/except`.

### 7. Test Results

Check:

- model vs naive baseline;
- model vs simple baseline;
- train-validation gap;
- confusion matrix for classification;
- residuals for regression;
- forecast horizon errors for time series;
- segment-level errors for business groups.

If a complex model barely beats a simple one, ship the simple one.

### 8. Finish

Final response format:

```text
Done: notebook runs through baseline -> tuned model.
Best: RandomForest, valid F1 0.81 vs baseline 0.62.
Params: max_depth=12, n_estimators=300.
Checks: split exclusivity, no null target, metric range.
Caveat: test set is small.
```

## Metric Selection

| Problem | Use |
|---|---|
| balanced binary classification | accuracy, ROC-AUC |
| imbalanced binary classification | PR-AUC, F1, recall, precision |
| multiclass balanced | accuracy, macro F1 |
| multiclass imbalanced | macro F1 |
| probability quality | log loss, calibration |
| regression business error | MAE |
| regression large-error penalty | RMSE |
| multiplicative/skewed regression | RMSLE |
| time series | MAE, RMSE, MAPE/sMAPE by horizon |
| clustering | stability and interpretability; silhouette as support |

Accuracy is not enough when the target is imbalanced.

## Examples

Detailed method coverage lives in:

- [Baseline Research: Machine Learning Methods](references/baseline-research.md)
- [GitHub Syntax References and Latest Research Tracker](references/github-and-latest-research.md)
- [Machine Learning Method Catalog](examples/method-catalog.md)
- [Classification Example](examples/classification.md)
- [Regression Example](examples/regression.md)
- [Time Series Example](examples/time-series.md)
- [NLP Example](examples/nlp.md)
- [Data Scraping Example](examples/data-scraping.md)

The method catalog covers logistic regression, KNN, Naive Bayes, decision trees, random forests, gradient boosting, SVM, linear/ridge/lasso/elastic net regression, clustering, PCA/t-SNE, time series, anomaly detection, NLP, computer vision, recommender systems, causal inference, neural networks, ensembles, and explainability. The NLP example expands text classification, topic modeling, document clustering, keyword extraction, semantic similarity, text regression, multilingual text, embeddings, and transformer upgrade rules. The scraping example covers API-first collection, static HTML, dynamic pages, downloadable files, and data quality checks.

### Classification

```text
Use $data-science-modeling to build a customer churn classifier.
Target: churn.
Metric: F1 because churn is imbalanced.
Output: notebook with baseline, logistic regression, one tuned tree model, checks, and short result.
```

Expected output:

```text
Best: LogisticRegression, valid F1 0.74 vs majority baseline 0.00.
Params: C=1, class_weight=balanced.
Checks: stratified split, no null target, no overlap, metric range.
```

### Regression

```text
Use $data-science-modeling to predict house prices.
Metric: MAE.
Compare median baseline, linear regression, and random forest.
```

Expected output:

```text
Best: RandomForestRegressor, valid MAE 18,420 vs median baseline 42,900.
Checks: target positive, split exclusive, predictions finite.
```

### Time Series

```text
Use $data-science-modeling to forecast monthly inflation.
Research comparable methodology first.
Use time-aware validation and compare against seasonal naive.
```

Expected output:

```text
Best: SARIMAX, validation MAE 0.18 vs seasonal naive 0.24.
Checks: chronological split, no future features, horizon-wise error.
```

### Notebook Debug

```text
Use $data-science-modeling to audit this notebook.
The validation score is 0.99 and seems suspicious.
```

Expected audit focus:

- target leakage;
- fitted preprocessing before split;
- duplicate rows across train/valid;
- target-derived features;
- wrong metric calculation;
- validation data used during feature selection.

## Anti-Patterns

Do not do this:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

Fit preprocessing after splitting:

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
```

Do not report only:

```text
Accuracy: 0.94
```

Report:

```text
Baseline F1: 0.00
Validation F1: 0.71
Precision: 0.66
Recall: 0.77
```

Do not tune before baseline:

```python
GridSearchCV(XGBClassifier(), huge_grid, cv=10)
```

Baseline first:

```python
DummyClassifier(strategy="most_frequent")
LogisticRegression(max_iter=1000)
```

## Rules

- Baseline before complexity.
- Split before fitted preprocessing.
- EDA decisions use training data only.
- Metric follows objective.
- Tuning stays small until evidence says otherwise.
- Notebook runs top-to-bottom.
- Every non-trivial transformation gets a small check.
- Report train vs validation when overfit is possible.
- Prefer installed libraries and standard patterns.
- Do not add dependencies for convenience.
- Do not claim "best" without defining the search space and validation design.

## File Layout

Local skill:

```text
~/.codex/skills/data-science-modeling/
├── SKILL.md
└── agents/
    └── openai.yaml
```

Suggested GitHub repo layout:

```text
data-science-modeling/
├── README.md
├── LICENSE
├── skills/
│   └── data-science-modeling/
│       └── SKILL.md
├── examples/
│   ├── classification.md
│   ├── regression.md
│   └── time-series.md
└── benchmarks/
    └── methodology.md
```

## Development

When changing the skill:

1. Edit `SKILL.md`.
2. Keep the description trigger-specific.
3. Keep the body short enough to load quickly.
4. Validate:

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  ~/.codex/skills/data-science-modeling
```

5. Forward-test on at least one classification, one regression, and one time-series task.

## FAQ

### Does this always browse Google Scholar?

No. Browse only when current methods, literature, benchmark, or domain methodology matter. If the task is a simple local notebook fix, skip browsing.

### Does this always use machine learning?

No. If descriptive analysis, SQL aggregation, or a simple statistical model answers the question, use that.

### Can it use XGBoost, LightGBM, or CatBoost?

Yes, if installed or justified. Start with a simple baseline first.

### Can it create production pipelines?

Yes, but only when requested. Default output is a clean notebook.

### Why not maximize accuracy?

Because accuracy can lie on imbalanced data. The metric must match the objective.

### Why so strict about leakage?

Because leakage creates impressive numbers and useless models.

## License

MIT. Use it, modify it, keep the models honest.

## Source Inspiration

Inspired by the documentation style of Ponytail:

https://github.com/DietrichGebert/ponytail

This project does not copy Ponytail's claims or benchmarks. It copies the useful documentation shape: clear premise, before/after, how it works, install, commands, rules, examples, FAQ, license.
