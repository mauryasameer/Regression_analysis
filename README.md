# Regression Analysis — Linear Regression Deep Dive

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

An in-depth exploration of linear regression — from raw data to production-grade model validation. Built on the Ames, Iowa housing dataset.

The companion Medium article walks through every section of the notebook narrative.

---

## What's inside

### Notebook: `notebooks/Regression_Analysis.ipynb`

A complete end-to-end regression study covering:

- **Exploratory Data Analysis** — distributions, correlations, outlier detection
- **Feature Engineering** — handling missing values, encoding, log-transformation of skewed targets
- **OLS Regression** — model fitting, coefficient interpretation
- **SHAP Attributions** — local feature importance for individual predictions
- **Model Risk Management (MRM) Diagnostics** — auditing OLS assumptions per SR 11-7:
  - Multicollinearity: Variance Inflation Factors (VIF)
  - Autocorrelation: Durbin-Watson statistic (2.03)
  - Heteroscedasticity: Breusch-Pagan test
  - Normality of Errors: Jarque-Bera test
  - Stability: Train vs test R² gap monitoring

### Training script: `scripts/train_model.py`

Reproducible pipeline using scikit-learn — fits the same model as the notebook, serialises the artifact to `src/data/`.

### MRM service: `src/services/mrm_service.py`

Standalone diagnostic module. Computes VIF, Durbin-Watson, Breusch-Pagan, and Jarque-Bera on any fitted OLS residuals.

---

## Dataset

[Ames, Iowa Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) — 2,919 residential property sales with 80 features.

Download and save to `src/data/train.csv` before running the notebook or training script.

---

## Quick Start

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the model
python scripts/train_model.py

# Open the notebook
jupyter notebook notebooks/Regression_Analysis.ipynb
```

---

## Repository Layout

```
.
├── notebooks/
│   └── Regression_Analysis.ipynb   # Main analysis notebook
├── scripts/
│   └── train_model.py              # Reproducible training script
├── src/
│   ├── services/
│   │   └── mrm_service.py          # MRM diagnostics (VIF, DW, BP, JB)
│   └── utils/
│       └── preprocessing.py        # scikit-learn transformers (Ames)
├── tests/
│   ├── integration/
│   │   └── test_model_pipeline.py
│   └── unit/
│       └── test_preprocessing.py
├── pyproject.toml                  # Ruff config
└── requirements.txt
```

---

## Related

**[housing-valuation-engine](https://github.com/mauryasameer/housing-valuation-engine)** — production ML app built on top of this work. Adds Delhi NCR predictions, a Streamlit dashboard, Folium map, and a monthly data refresh pipeline.

---

## Quality

```bash
ruff check .
pytest
```
