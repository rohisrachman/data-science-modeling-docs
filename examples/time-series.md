# Time Series Example

Prompt:

```text
Use $data-science-modeling to forecast monthly inflation.
Research comparable methodology first.
Use chronological validation.
Compare seasonal naive, moving average, and one statistical/ML model.
```

Expected notebook sections:

1. Setup
2. Load or collect data
3. Data source/data card if scraped
4. Date index audit
5. Chronological split
6. Train-only decomposition/EDA
7. Seasonal naive baseline
8. Candidate model
9. Backtest by horizon
10. Error analysis
11. Final result

Minimum checks:

```python
assert df.index.is_monotonic_increasing
assert df.index.is_unique
assert train.index.max() < valid.index.min()
assert forecast.index.min() >= valid.index.min()
assert np.isfinite(valid_mae)
```

Final response pattern:

```text
Best: SARIMAX, validation MAE 0.18 vs seasonal naive 0.24.
Checks: chronological split, no future features, horizon-wise MAE.
Caveat: shock months dominate the error.
```
