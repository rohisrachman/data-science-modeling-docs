# NLP Example

Use this when `$data-science-modeling` handles text data: sentiment analysis, topic modeling, text classification, keyword extraction, document clustering, similarity search, summarization evaluation, or text regression.

The default rule still applies: start with a simple baseline before embeddings, transformers, or fine-tuning.

## Method Map

| NLP task | Baseline | Upgrade |
|---|---|---|
| sentiment/text classification | majority + TF-IDF logistic regression | LinearSVC, embeddings, transformer fine-tune |
| multiclass topic/category classification | majority + TF-IDF linear model | calibrated SVM, transformer |
| topic modeling | TF-IDF + NMF | LDA, BERTopic/embeddings if installed |
| document clustering | TF-IDF + k-means | embeddings + clustering |
| keyword extraction | TF-IDF top terms | KeyBERT/embedding method if installed |
| semantic similarity | TF-IDF cosine | sentence embeddings |
| text regression | median + TF-IDF ridge | embeddings + gradient boosting/MLP |
| sequence labeling | rule baseline | CRF/transformer if required |

## Text Classification

Prompt:

```text
Use $data-science-modeling to build an NLP text classification notebook.
Task: sentiment classification.
Text column: review_text.
Target: sentiment.
Metric: macro F1.
Start with majority baseline, then TF-IDF + LogisticRegression and LinearSVC.
Check duplicate text across splits and inspect top errors.
```

Minimal notebook flow:

1. Load data
2. Audit text column and target
3. Remove or flag exact duplicates
4. Stratified split before vectorization
5. Majority baseline
6. TF-IDF + logistic regression
7. TF-IDF + LinearSVC
8. Error analysis
9. Short result

Minimal syntax:

```python
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
TEXT = "review_text"
TARGET = "sentiment"

assert TEXT in df.columns
assert TARGET in df.columns
assert df[TEXT].notna().all()
assert df[TARGET].notna().all()

df = df.copy()
df[TEXT] = df[TEXT].astype(str).str.strip()
df = df[df[TEXT].str.len() > 0]

X_train, X_valid, y_train, y_valid = train_test_split(
    df[TEXT],
    df[TARGET],
    test_size=0.2,
    stratify=df[TARGET],
    random_state=RANDOM_STATE,
)

assert set(X_train.index).isdisjoint(set(X_valid.index))
assert set(X_train).isdisjoint(set(X_valid))  # exact duplicate leakage check

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_valid)

logreg = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=50000,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_valid)

svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=50000,
    )),
    ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
])

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_valid)

scores = pd.DataFrame([
    {"model": "majority", "macro_f1": f1_score(y_valid, baseline_pred, average="macro")},
    {"model": "tfidf_logreg", "macro_f1": f1_score(y_valid, logreg_pred, average="macro")},
    {"model": "tfidf_linearsvc", "macro_f1": f1_score(y_valid, svm_pred, average="macro")},
]).sort_values("macro_f1", ascending=False)

scores
```

Error analysis:

```python
errors = pd.DataFrame({
    "text": X_valid,
    "actual": y_valid,
    "pred": svm_pred,
})

errors = errors[errors["actual"] != errors["pred"]]
errors.sample(min(10, len(errors)), random_state=RANDOM_STATE)
```

Final response pattern:

```text
Best: TF-IDF + LinearSVC, valid macro F1 0.82 vs majority baseline 0.31.
Checks: non-null text, stratified split, exact duplicate leakage check, macro F1.
Caveat: sarcasm and mixed-language reviews dominate errors.
```

## Topic Modeling

Prompt:

```text
Use $data-science-modeling to run topic modeling on news articles.
Text column: article_text.
Start with TF-IDF + NMF.
Test 5, 8, and 12 topics.
Show top terms and sample documents per topic.
```

Minimal syntax:

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT = "article_text"

docs = df[TEXT].dropna().astype(str).str.strip()
docs = docs[docs.str.len() > 0]

vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.85,
    ngram_range=(1, 2),
    max_features=30000,
)
X_text = vectorizer.fit_transform(docs)

nmf = NMF(n_components=8, random_state=42, max_iter=500)
W = nmf.fit_transform(X_text)
H = nmf.components_

terms = np.array(vectorizer.get_feature_names_out())

for topic_id, weights in enumerate(H):
    top_terms = terms[weights.argsort()[-12:]][::-1]
    print(f"Topic {topic_id}: {', '.join(top_terms)}")
```

Assign dominant topic:

```python
topic_id = W.argmax(axis=1)
topic_strength = W.max(axis=1)

topic_df = pd.DataFrame({
    "text": docs.values,
    "topic": topic_id,
    "strength": topic_strength,
})

topic_df.groupby("topic").size().sort_values(ascending=False)
```

Checks:

- inspect top terms manually;
- inspect sample documents per topic;
- do not treat topics as ground truth;
- rerun with a few topic counts.

Final response pattern:

```text
Topic model built with TF-IDF + NMF.
Best reviewed setting: 8 topics.
Checks: empty text removed, top terms inspected, sample docs per topic reviewed.
Caveat: topic labels are analyst-assigned, not model truth.
```

## Document Clustering

Prompt:

```text
Use $data-science-modeling to cluster complaint texts.
Start with TF-IDF + k-means.
Compare k=5..12 using silhouette and manual topic interpretability.
```

Minimal syntax:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

results = []
for k in range(5, 13):
    vec = TfidfVectorizer(min_df=5, max_df=0.85, ngram_range=(1, 2), max_features=30000)
    X_text = vec.fit_transform(docs)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_text)
    score = silhouette_score(X_text, labels)
    results.append({"k": k, "silhouette": score})

pd.DataFrame(results)
```

Checks:

- silhouette is only support;
- cluster labels must be interpretable;
- sample documents per cluster must make sense.

## Keyword Extraction

Prompt:

```text
Use $data-science-modeling to extract keywords from article_text.
Use TF-IDF first. Return top corpus keywords and top keywords per document.
```

Minimal syntax:

```python
vec = TfidfVectorizer(
    lowercase=True,
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2),
    stop_words="english",
)
X_text = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())

mean_tfidf = np.asarray(X_text.mean(axis=0)).ravel()
top_idx = mean_tfidf.argsort()[-30:][::-1]
terms[top_idx]
```

Checks:

- stopwords match language;
- remove boilerplate text;
- inspect noisy terms.

## Semantic Similarity

Prompt:

```text
Use $data-science-modeling to find similar documents.
Start with TF-IDF cosine similarity.
Only use embeddings if an embedding model/library already exists in the project.
```

Minimal syntax:

```python
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000)
X_text = vec.fit_transform(docs)

query = ["inflation expectations and food prices"]
q = vec.transform(query)
sim = cosine_similarity(q, X_text).ravel()

top = sim.argsort()[-10:][::-1]
docs.iloc[top]
```

Checks:

- evaluate with known relevant examples if available;
- inspect top results manually;
- explain lexical vs semantic limitations.

## Text Regression

Prompt:

```text
Use $data-science-modeling to predict review rating from review_text.
Metric: MAE.
Compare median baseline, TF-IDF + Ridge, and optional tree/boosting model.
```

Minimal syntax:

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

baseline = DummyRegressor(strategy="median")
baseline.fit(X_train_text.to_frame(), y_train)

ridge_text = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000)),
    ("model", Ridge(alpha=1.0)),
])
ridge_text.fit(X_train_text, y_train)

baseline_mae = mean_absolute_error(y_valid, baseline.predict(X_valid_text.to_frame()))
ridge_mae = mean_absolute_error(y_valid, ridge_text.predict(X_valid_text))
baseline_mae, ridge_mae
```

Checks:

- target numeric and finite;
- duplicates across split checked;
- prediction range sanity checked.

## Multilingual Text

Prompt:

```text
Use $data-science-modeling for Indonesian-English mixed sentiment analysis.
Do not use English stopwords blindly. Audit language mix first.
```

Checks:

- identify language distribution;
- avoid wrong stopword list;
- inspect tokenization on samples;
- consider character n-grams for noisy/mixed text.

Minimal vectorizer for noisy multilingual text:

```python
char_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ("clf", LinearSVC(class_weight="balanced", random_state=42)),
])
```

## Transformer Or Embedding Upgrade

Use embeddings or transformers only when:

- TF-IDF baseline underperforms;
- semantic meaning matters;
- enough labeled data exists;
- the required package/model is already available or explicitly approved;
- runtime is acceptable.

Prompt:

```text
Use $data-science-modeling to compare TF-IDF baseline with existing sentence embeddings.
Do not install new packages. Use macro F1 and inspect errors.
```

Checks:

- baseline comparison included;
- embedding generation does not use labels;
- duplicate leakage checked;
- model version documented.

## NLP Anti-Patterns

Do not vectorize before splitting:

```python
X = vectorizer.fit_transform(df[TEXT])
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
```

Do this:

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    df[TEXT], df[TARGET], stratify=df[TARGET], random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
```

Do not report accuracy only for imbalanced sentiment:

```text
Accuracy: 0.91
```

Report:

```text
Macro F1: 0.72
Positive recall: 0.68
Negative recall: 0.80
Baseline macro F1: 0.33
```

## Final NLP Result Template

```text
Best: TF-IDF + LinearSVC
Validation macro F1: 0.82
Baseline macro F1: 0.31
Checks: non-null text, duplicate leakage, stratified split, macro F1
Main errors: sarcasm, mixed-language slang, very short texts
Next: threshold/calibration or embeddings only if semantic errors dominate
```
