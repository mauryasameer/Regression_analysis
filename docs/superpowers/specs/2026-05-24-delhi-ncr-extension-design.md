# Delhi NCR House Price Predictor — Design Spec

**Date:** 2026-05-24
**Status:** Approved

---

## Overview

Extend the existing Real Estate ML Portal to support Delhi NCR flat price prediction and rent estimation alongside the existing Ames, Iowa model. Phase 1 covers flats only across five NCR regions: Gurgaon, Noida, Delhi, Faridabad, Ghaziabad. Gurgaon, Noida, and Delhi ship first with trained models; Faridabad and Ghaziabad unlock as data becomes available.

Each region supports two prediction modes — **Buy** (sale price in INR) and **Rent** (monthly rent in INR) — using separate trained models per region per mode.

---

## Architecture

### App structure

A tab switcher at the top of `app.py` routes between two independent prediction experiences:
- **🇺🇸 Ames, Iowa** — existing, unchanged
- **🇮🇳 Delhi NCR** — new tab, Layout C UI

No changes to existing Ames pipeline, services, or preprocessing code.

### New files

```
scripts/
  train_delhi_ncr.py          # per-region training, saves .joblib artifacts
  scrape_delhi_ncr.py         # fetches listings from 99acres per region

src/
  utils/delhi_preprocessing.py     # Delhi-specific feature engineering pipeline
  services/delhi_mrm_service.py    # MRM diagnostics (wraps existing mrm_service logic)

configs/
  delhi_ncr_regions.yaml      # source of truth: which regions have trained models

src/data/delhi_ncr/           # gitignored — raw CSVs and training data only
  {region}.csv

models/delhi_ncr/             # tracked — committed .joblib artifacts (safe to commit, no PII)
  model_{region}_sale.joblib
  model_{region}_rent.joblib
  metadata_{region}_sale.joblib
  metadata_{region}_rent.joblib
  shap_{region}_sale.joblib
  shap_{region}_rent.joblib

.github/workflows/
  delhi_data_refresh.yml      # monthly cron: scrape → retrain → commit

docs/superpowers/specs/
  2026-05-24-delhi-ncr-extension-design.md  # this file

tests/
  unit/test_delhi_preprocessing.py
  unit/test_scraper.py
  integration/test_delhi_pipeline.py
  test_data/delhi_fixture.csv               # 5-row fixture for CI
```

### Region config (`configs/delhi_ncr_regions.yaml`)

```yaml
regions:
  - name: Gurgaon
    lat: 28.4595
    lng: 77.0266
    model_ready: true
    localities: [DLF Phase 1, DLF Phase 2, Sector 56, Golf Course Road, Sohna Road]
  - name: Noida
    lat: 28.5355
    lng: 77.3910
    model_ready: true
    localities: [Sector 18, Sector 62, Sector 137, Sector 150, Greater Noida]
  - name: Delhi
    lat: 28.6139
    lng: 77.2090
    model_ready: true
    localities: [Dwarka, Rohini, Saket, Vasant Kunj, Lajpat Nagar, Janakpuri]
  - name: Faridabad
    lat: 28.4089
    lng: 77.3178
    model_ready: false
    localities: []
  - name: Ghaziabad
    lat: 28.6692
    lng: 77.4538
    model_ready: false
    localities: []
```

Adding a new region = add one entry here + drop a new `.joblib`. No UI code changes required.

---

## Data Pipeline

### One-time bootstrap

1. Download [Delhi NCR Housing dataset from Kaggle](https://www.kaggle.com/datasets/goelyash/housing-price-dataset-of-delhincr) → `src/data/delhi_ncr/base.csv`
2. Run `python scripts/train_delhi_ncr.py` — splits by region, trains one Lasso model per ready region, saves artifacts to `src/data/delhi_ncr/`

### Monthly refresh (GitHub Actions)

Cron: `0 20 1 * *` (1st of month, 2am IST = 8:30pm UTC previous day)

Steps:
1. `scrape_delhi_ncr.py` hits 99acres public listings per region
2. Deduplicates by `sha256(locality + area + floor + price)`
3. Appends new rows to `src/data/delhi_ncr/{region}.csv`
4. Retrains only regions with ≥50 new rows since last training
5. Commits updated `.joblib` artifacts to `models/delhi_ncr/` on a `data/refresh-YYYY-MM` branch
6. Opens a PR to `dev` automatically — merge is manual (one-click)
7. On failure: auto-opens a GitHub issue with error details

### Features

| Feature | Type | Notes |
|---|---|---|
| `bhk` | int | 1–5 |
| `area_sqft` | float | |
| `floor` | int | |
| `total_floors` | int | |
| `age_years` | float | Current year − build year |
| `furnishing` | categorical | Furnished / Semi-Furnished / Unfurnished |
| `locality` | categorical | One-hot encoded per region |
| `parking` | bool | |
| `lift` | bool | |
| `metro_dist_km` | float | Distance to nearest metro station |

**Targets:**
- `price_inr` — sale price, log-transformed (`np.log1p`)
- `rent_inr` — monthly rent, log-transformed (`np.log1p`)

Two separate LassoCV models per region (one for sale, one for rent), each wrapped in `TransformedTargetRegressor`. Training data sourced separately for each mode from the Kaggle dataset and scraper.

---

## UI — Delhi NCR Tab

**Layout C: Sidebar + Map + Form**

Three-panel layout within the Delhi NCR tab:

| Panel | Content |
|---|---|
| Left sidebar (150px) | Region list — green dot = model ready, grey = coming soon. Click to select. |
| Centre | Leaflet map (CartoDB dark tiles). Green pins = ready, grey = coming soon. Click pin → popup with "Select region" button. Selecting a region flies the map to that region. |
| Right form (240px) | **Buy / Rent toggle** at top. Prediction form — region badge, BHK, area, floor, total floors, locality dropdown (updates per region), age, furnishing, parking toggle, lift toggle, metro distance. Predict button → shows estimated sale price or monthly rent in INR with confidence range. SHAP waterfall below. |

Region availability on map and sidebar driven entirely by `configs/delhi_ncr_regions.yaml`.

Leaflet loaded via CDN (`unpkg.com/leaflet@1.9.4`). No additional npm/bundler setup.

---

## Monthly Refresh Workflow

```yaml
# .github/workflows/delhi_data_refresh.yml
on:
  schedule:
    - cron: "0 20 1 * *"
  workflow_dispatch:        # manual trigger available

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - setup python 3.12
      - pip install -r requirements.txt
      - run scrape_delhi_ncr.py
      - run train_delhi_ncr.py --only-updated
      - commit models/delhi_ncr/ to data/refresh-YYYY-MM branch
      - open PR to dev if artifacts changed
      - open issue on failure
```

---

## Testing

| File | What it tests |
|---|---|
| `tests/unit/test_delhi_preprocessing.py` | Feature engineering: correct one-hot encoding, age calculation, missing value handling |
| `tests/unit/test_scraper.py` | Scraper output schema — stubbed HTTP, no live calls in CI |
| `tests/integration/test_delhi_pipeline.py` | End-to-end inference on `tests/test_data/delhi_fixture.csv` — 5 representative rows |

All tests follow existing project standards: `pytest-mock` for patching, assert on output shape not internal calls, no live HTTP in CI.

---

## Out of Scope (Phase 1)

- Property types other than flats (builder floors, villas, plots)
- Historical price trend charts
- User accounts or saved predictions
