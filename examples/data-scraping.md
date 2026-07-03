# Data Scraping Example

Use this when `$data-science-modeling` needs to collect data before analysis or modeling.

Scraping is data engineering before it is data science. A model built on brittle, duplicated, or legally questionable scraped data is not a good model.

## Rule

Prefer the least fragile data source:

1. official dataset/download;
2. official API;
3. public static HTML;
4. rendered browser automation;
5. manual collection.

Do not scrape private, authenticated, paywalled, personal, or restricted data unless the user confirms they have permission.

## Scraping Workflow

1. Define the data target.
   - fields needed;
   - time range;
   - geography/entity scope;
   - update frequency;
   - expected row count.

2. Check source legitimacy.
   - official API or data portal first;
   - terms/robots/rate limits where applicable;
   - avoid credentials and private areas unless explicitly authorized.

3. Inspect page type.
   - static HTML: `requests` + parser;
   - embedded JSON: parse script data;
   - dynamic JS: browser automation only if needed;
   - file download: CSV/XLSX/PDF parser.

4. Build minimal extractor.
   - fetch;
   - parse;
   - normalize;
   - validate schema;
   - save raw and cleaned data separately if the project needs reproducibility.

5. Test data quality.
   - row count;
   - null rate;
   - duplicate keys;
   - date parsing;
   - numeric parsing;
   - source timestamp;
   - sample records.

6. Finish with a short data card.
   - source;
   - date collected;
   - rows/columns;
   - known caveats;
   - next refresh command if any.

## Static HTML Scraping

Prompt:

```text
Use $data-science-modeling to scrape a public static table for inflation data.
Prefer official download/API first. If scraping HTML, save raw HTML and cleaned CSV.
Validate date, value, duplicates, and row count.
```

Minimal syntax:

```python
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://example.com/public-table"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "data-research-contact@example.com"}
resp = requests.get(URL, headers=headers, timeout=30)
resp.raise_for_status()

raw_path = RAW_DIR / "source.html"
raw_path.write_text(resp.text, encoding="utf-8")

soup = BeautifulSoup(resp.text, "html.parser")
table = soup.select_one("table")
assert table is not None, "table not found"

df = pd.read_html(str(table))[0]
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

assert len(df) > 0
assert df.columns.is_unique

df.to_csv(OUT_DIR / "scraped_table.csv", index=False)
df.head()
```

Checks:

```python
assert df.columns.is_unique
assert not df.duplicated().any()
assert len(df) >= 10
```

## API-First Collection

Prompt:

```text
Use $data-science-modeling to collect data from the official API.
Handle pagination, rate limits, schema validation, and save raw JSON plus cleaned CSV.
```

Minimal syntax:

```python
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.example.com/v1/items"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
page = 1

while True:
    resp = requests.get(BASE_URL, params={"page": page, "per_page": 100}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    items = payload.get("data", [])
    if not items:
        break

    rows.extend(items)
    if not payload.get("has_next", False):
        break

    page += 1
    time.sleep(0.5)

df = pd.json_normalize(rows)
assert len(df) > 0
assert df.columns.is_unique

df.to_json(RAW_DIR / "api_rows.json", orient="records", indent=2)
df.to_csv(OUT_DIR / "api_rows.csv", index=False)
```

Checks:

- no silent pagination truncation;
- HTTP errors fail loudly;
- rate limits respected;
- schema drift detected.

## Dynamic Page Scraping

Use Playwright/Selenium only when the data is not available through API, file download, static HTML, or embedded JSON.

Prompt:

```text
Use $data-science-modeling to scrape a public dynamic page.
First check whether the data is in an API request or embedded JSON.
Use browser automation only if necessary.
```

Minimal Playwright-style pattern:

```python
from playwright.sync_api import sync_playwright

URL = "https://example.com/public-dynamic-page"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, wait_until="networkidle", timeout=60_000)

    rows = page.locator("table tbody tr").all()
    data = []
    for row in rows:
        cells = [c.inner_text().strip() for c in row.locator("td").all()]
        data.append(cells)

    browser.close()

df = pd.DataFrame(data)
assert len(df) > 0
```

Checks:

- stable selector;
- wait condition explicit;
- screenshot or saved HTML when debugging;
- rate limited and respectful.

## PDF, XLSX, CSV Downloads

Prompt:

```text
Use $data-science-modeling to collect official downloadable data.
Prefer CSV/XLSX over scraping HTML. Keep the original file and produce a cleaned CSV.
```

Minimal syntax:

```python
from pathlib import Path

import pandas as pd
import requests

URL = "https://example.com/data.xlsx"
RAW = Path("data/raw/data.xlsx")
OUT = Path("data/processed/data.csv")
RAW.parent.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

resp = requests.get(URL, timeout=60)
resp.raise_for_status()
RAW.write_bytes(resp.content)

df = pd.read_excel(RAW)
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
assert len(df) > 0

df.to_csv(OUT, index=False)
```

## Data Quality Contract

Every scraped dataset should have:

```python
data_card = {
    "source_url": URL,
    "collected_at": pd.Timestamp.utcnow().isoformat(),
    "rows": len(df),
    "columns": list(df.columns),
    "duplicate_rows": int(df.duplicated().sum()),
    "null_rate": df.isna().mean().round(4).to_dict(),
}

data_card
```

Minimum checks:

```python
assert len(df) > 0
assert df.columns.is_unique
assert df.duplicated().mean() < 0.05
```

For modeling:

- define the observation unit;
- define the target creation rule;
- avoid future information in scraped features;
- timestamp the collection;
- keep raw source data if refresh/reproducibility matters.

## Anti-Patterns

Do not:

- scrape when an official API/download exists;
- ignore terms, robots, or rate limits;
- use brittle absolute XPath selectors as the first choice;
- overwrite raw data without versioning;
- silently swallow HTTP/parser errors;
- train a model before checking duplicates and schema drift;
- scrape private/personal data without permission.

## Final Response Template

```text
Done: scraped public source into data/processed/source.csv.
Rows: 2,418. Columns: 12.
Checks: HTTP status, non-empty table, unique columns, duplicate rate 0.3%, parsed date column.
Caveat: source page has no official update timestamp.
Next: run modeling notebook using data/processed/source.csv.
```
