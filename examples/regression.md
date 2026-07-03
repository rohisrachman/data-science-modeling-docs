# Regression Example

Prompt:

```text
Use $data-science-modeling to build a price prediction notebook.
Target: price.
Metric: MAE.
Compare median baseline, linear regression, ridge, and one tree model.
```

Expected notebook sections:

1. Setup
2. Load or collect data
3. Data source/data card if scraped
4. Target and outlier audit
5. Split
6. Train-only EDA
7. Preprocessing pipeline
8. Median baseline
9. Linear/ridge baseline
10. Candidate tree model
11. Error analysis by segment
12. Final test evaluation

Minimum checks:

```python
assert TARGET in df.columns
assert df[TARGET].notna().all()
assert df.columns.is_unique
assert np.isfinite(y_train).all()
assert np.isfinite(valid_mae)
```

Final response pattern:

```text
Best: Ridge, valid MAE 18,900 vs median baseline 42,100.
Params: alpha=10.
Checks: target finite, split exclusive, prediction shape, metric finite.
Caveat: high-end prices are underpredicted.
```
