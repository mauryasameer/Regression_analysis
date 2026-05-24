# Secure Housing Valuation Engine 🏠

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](#)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A production-grade real estate ML portal with two prediction experiences:

- **🇺🇸 Ames, Iowa** — log-transformed LassoCV model with SHAP attributions, MRM diagnostics, and local LLM audit narratives
- **🇮🇳 Delhi NCR** — flat sale price and monthly rent prediction for Gurgaon, Noida, and Delhi, with an interactive Folium map and Buy/Rent toggle

Both models share the same scikit-learn pipeline architecture and Model Risk Management layer. Raw data never leaves the machine.

---

## 🏗️ System Architecture

Both prediction pipelines share the same architecture pattern:

```
                  ┌────────────────────────────────────────┐
                  │          Raw House Parameters          │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    scikit-learn Preprocessing Pipeline  │ (Data-leakage prevention)
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Target-Transformed Lasso Regressor  │ (Inference in actual dollars)
                  └─────────┬────────────────────┬─────────┘
                            │                    │
                            ▼                    ▼
     ┌──────────────────────────────┐    ┌──────────────────────────────┐
     │    SHAP Local Attributions   │    │  Model Risk Governance Audit │ (VIF, Normality,
     └──────────────┬───────────────┘    └──────────────────────────────┘  Heteroscedasticity)
                    │
                    ▼
     ┌──────────────────────────────┐
     │    Local Cybersecurity Gate  │ (Input sanitization & injection check)
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Local LLM (Ollama/Llama3.1) │ (No Data Egress / Zero Trust Privacy)
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Human-Readable Valuation   │
     │      Audit Narrative         │
     └──────────────────────────────┘
```

---

## 🛠️ Core Features

### 1. Leak-Proof ML Pipeline
* Categorical filling, feature engineering, and scaling encapsulated in a scikit-learn `Pipeline`.
* Statistics calculated exclusively on training data — zero leakage into test folds or inference rows.
* `TransformedTargetRegressor` manages log-transformation of targets (`SalePrice` / `price_inr` / `rent_inr`).

### 2. Delhi NCR Flat Price Predictor
* **Regions:** Gurgaon, Noida, Delhi (model ready); Faridabad, Ghaziabad (coming soon)
* **Modes:** Buy (sale price in INR) and Rent (monthly rent in INR) — separate LassoCV models per region per mode
* **UI:** Layout C — sidebar region list + Folium map with clickable pins + prediction form with Buy/Rent toggle
* **Features:** BHK, area, floor, age, furnishing, locality, parking, lift, metro distance
* **Monthly refresh:** GitHub Actions cron scrapes 99acres, retrains updated regions, opens auto-PR to `dev`

To activate Delhi NCR predictions:
```bash
# Download dataset from Kaggle → src/data/delhi_ncr/base.csv, then:
python scripts/train_delhi_ncr.py
```

### 3. Model Risk Management (MRM) Diagnostics
To comply with model validation guidelines (such as SR 11-7), the application audits OLS regression assumptions:
* **Multicollinearity**: Variance Inflation Factors (VIF)
* **Autocorrelation**: Durbin-Watson statistic (Ames: **2.03**)
* **Heteroscedasticity**: Breusch-Pagan test
* **Normality of Errors**: Jarque-Bera test
* **Stability Audit**: Train vs test R² gap monitoring

### 4. Local-First AI Privacy & Cybersecurity (Ames tab)
* **Data Isolation**: Local LLM via **Ollama** (`llama3.1:8b`). No data egress to cloud APIs.
* **Injection Safeguards**: Regex-based prompt injection detection on all text inputs.
* **Output Sanitization**: HTML script/style tag stripping to prevent XSS.

---

## 📂 Repository Layout

```
.
├── src/
│   ├── core/
│   │   └── interfaces.py            # Decoupled LLMProvider interface
│   ├── providers/
│   │   └── ollama_provider.py       # Concrete local Ollama provider
│   ├── services/
│   │   ├── explanation_service.py   # Prompt structuring & LLM coordination
│   │   ├── mrm_service.py           # Statistical diagnostic audits (Ames)
│   │   └── delhi_mrm_service.py     # MRM re-export for Delhi NCR tab
│   ├── utils/
│   │   ├── preprocessing.py         # Custom scikit-learn transformers (Ames)
│   │   ├── delhi_preprocessing.py   # DelhiFeatureTransformer, DelhiColumnFinalizer
│   │   └── security.py              # Input sanitization and prompt injection checks
│   └── data/                        # (Git-ignored) Model pipeline & raw CSVs
├── models/
│   └── delhi_ncr/                   # Committed .joblib artifacts (sale + rent per region)
├── configs/
│   └── delhi_ncr_regions.yaml       # Region config — drives sidebar, map, model loading
├── scripts/
│   ├── train_model.py               # Ames Iowa model training
│   ├── train_delhi_ncr.py           # Delhi NCR per-region sale + rent training
│   └── scrape_delhi_ncr.py          # 99acres scraper with sha256 deduplication
├── .github/workflows/
│   ├── ci.yml                       # Lint + test on Python 3.11 & 3.12
│   └── delhi_data_refresh.yml       # Monthly cron: scrape → retrain → auto-PR
├── tests/
│   ├── unit/
│   │   ├── test_delhi_preprocessing.py
│   │   └── test_scraper.py
│   ├── integration/
│   │   └── test_delhi_pipeline.py   # Skips when artifacts absent
│   └── test_data/
│       └── delhi_fixture.csv
├── notebooks/
│   └── Regression_Analysis.ipynb
├── docs/superpowers/
│   ├── specs/                       # Design specs
│   └── plans/                       # Implementation plans
├── app.py                           # Streamlit dashboard (Ames + Delhi NCR tabs)
├── pyproject.toml                   # Ruff config
├── requirements.txt                 # Pinned dependencies
├── VERSION                          # Version string
└── CHANGELOG.md                     # Semantic release history
```

---

## 🚀 Quick Start

### 1. Installation
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Ames Iowa model
```bash
python scripts/train_model.py
```

### 3. Train Delhi NCR models (optional)
Download the [Delhi NCR Housing dataset from Kaggle](https://www.kaggle.com/datasets/goelyash/housing-price-dataset-of-delhincr) and save it to `src/data/delhi_ncr/base.csv`, then:
```bash
python scripts/train_delhi_ncr.py
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. LLM audit narratives (optional, Ames tab only)
Install [Ollama](https://ollama.com/) and pull the model:
```bash
ollama run llama3.1:8b
```

---

## 🧪 Quality and Linting
To check code compliance, format styling, and import organization, run the Ruff linter:
```bash
ruff check .
```
All code files under `src/` and `app.py` are fully lint-compliant.