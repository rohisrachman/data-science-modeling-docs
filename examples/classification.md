# Classification Example

Prompt:

```text
Use $data-science-modeling to build a churn classification notebook.
Target: churn.
Metric: F1 because the positive class is rare.
Compare majority baseline, logistic regression, and one tree-based model.
Tune only if the simple model beats baseline.
```

Expected notebook sections:

1. Setup
2. Load or collect data
3. Data source/data card if scraped
4. Target audit
5. Stratified split
6. Train-only EDA
7. Preprocessing pipeline
8. Majority baseline
9. Logistic regression baseline
10. Candidate model
11. Small tuning
12. Error analysis
13. Final result

Minimum checks:

```python
assert TARGET in df.columns
assert df[TARGET].notna().all()
assert df.columns.is_unique
assert set(X_train.index).isdisjoint(set(X_valid.index))
assert 0 <= valid_f1 <= 1
```

Final response pattern:

```text
Best: LogisticRegression, valid F1 0.74 vs majority baseline 0.00.
Params: C=1, class_weight=balanced.
Checks: stratified split, no null target, split exclusivity, metric range.
Caveat: recall is stronger than precision; threshold tuning may be needed.
```
