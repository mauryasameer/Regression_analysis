# 🏠 Regression Analysis — Linear Regression Deep Dive

[![CI](https://github.com/mauryasameer/Regression_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/mauryasameer/Regression_analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange.svg)](notebooks/Regression_Analysis.ipynb)

> A production-grade study of linear regression on the **Ames, Iowa housing dataset** — from raw data exploration to model risk management diagnostics used in financial risk frameworks (SR 11-7).

---

## What This Covers

```
Raw Data (2,919 sales · 80 features)
        │
        ▼
┌─────────────────────────┐
│   Data Preprocessing    │  Missing value imputation · outlier removal
│   & Feature Engineering │  Encoding · derived columns · log-transform
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Model Training        │  Ridge · Lasso · OLS
│   (scikit-learn)        │  Cross-validated alpha selection
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   SHAP Attributions     │  Local feature importance per prediction
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   MRM Diagnostics       │  VIF · Durbin-Watson (2.03) · Breusch-Pagan
│   (SR 11-7 aligned)     │  Jarque-Bera · Train/Test stability gap
└─────────────────────────┘
```

---

## What You'll Learn

| Topic | Concept |
|-------|---------|
| **Preprocessing** | Handling 80-feature datasets with mixed types, nulls, and outliers |
| **Feature Engineering** | Derived columns, dummy encoding, log-transforming skewed targets |
| **Ridge vs Lasso** | Why L1 sparsity matters — automatic feature selection vs shrinkage |
| **Residual Diagnostics** | Validating OLS assumptions before trusting your model |
| **MRM / SR 11-7** | How model risk governance applies to regression models |
| **SHAP** | Explaining individual predictions, not just global feature importance |

---

## Key Results

| Model | Best α | Characteristic |
|-------|--------|----------------|
| Ridge | CV-selected | Shrinks all coefficients — retains every feature |
| Lasso | 0.0001 | Sparse — drives irrelevant coefficients to exactly zero |

**Residual health (Lasso model):**
- Durbin-Watson: **2.03** — no autocorrelation
- Residuals approximately normally distributed (Jarque-Bera)
- No significant heteroscedasticity (Breusch-Pagan)

---

## Quick Start

```bash
# 1. Set up environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset
# https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
# → save to src/data/train.csv

# 3. Train the model
python scripts/train_model.py

# 4. Open the notebook
jupyter notebook notebooks/Regression_Analysis.ipynb
```

---

## Repository Layout

```
.
├── notebooks/
│   └── Regression_Analysis.ipynb   # Main analysis — start here
├── scripts/
│   └── train_model.py              # Reproducible scikit-learn training script
├── src/
│   ├── services/
│   │   └── mrm_service.py          # MRM diagnostics (VIF, DW, BP, JB)
│   └── utils/
│       └── preprocessing.py        # Custom scikit-learn transformers
├── tests/
│   ├── integration/
│   │   └── test_model_pipeline.py
│   └── unit/
│       └── test_preprocessing.py
├── pyproject.toml                  # Ruff lint config
└── requirements.txt
```

---

## Related Project

**[housing-valuation-engine](https://github.com/mauryasameer/housing-valuation-engine)** — production ML app built on this work. Extends to Delhi NCR price prediction with GradientBoosting, a Streamlit dashboard, Folium map, and a monthly data refresh pipeline.

---

## Code Quality

```bash
ruff check .   # linting
pytest         # tests
```
