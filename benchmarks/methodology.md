# Benchmark Methodology

Do not publish performance claims for `data-science-modeling` without running a fair benchmark.

## Goal

Measure whether the skill improves data science outputs without hiding risk.

## Tasks

Use 8-12 realistic tasks:

- 3 classification notebooks;
- 3 regression notebooks;
- 2 time-series notebooks;
- 1 SQL-heavy feature engineering task;
- 1 public data scraping/API collection task;
- 1 notebook debugging task.

## Arms

Run the same agent in two modes:

1. no skill;
2. with `$data-science-modeling`.

Use the same prompt, dataset, environment, and time budget.

## Scorecard

| Area | Pass condition |
|---|---|
| runnable notebook | runs top-to-bottom |
| baseline | naive/simple baseline exists |
| leakage safety | split before fitted preprocessing |
| metric choice | metric matches objective |
| validation design | split matches data structure |
| tuning discipline | search space is justified and small |
| result reporting | baseline comparison and best model shown |
| code checks | assertions or tests exist |
| data acquisition | source, collection time, schema, duplicate/null checks documented when data is scraped |

## Metrics

Track:

- notebook execution success;
- number of leakage issues;
- number of dependencies added;
- LOC/cell count;
- runtime;
- validation/test metric vs baseline;
- scraped data quality checks where applicable;
- human review score.

## Reporting

Report failures as prominently as improvements. A shorter notebook with leakage is a failed notebook. A model trained on unvalidated scraped data is also a failed notebook.
