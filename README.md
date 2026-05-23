# Secure Housing Valuation Engine 🏠

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

This repository contains a modernized, production-grade housing valuation application that leverages log-transformed Lasso regularization, **SHAP (SHapley Additive exPlanations)** attributions, and local **Generative AI (Llama 3.1 8B)** to deliver secure, explainable real estate predictions.

Originally an experimental research notebook, this project was upgraded to comply with strict software engineering, **Model Risk Management (MRM)**, and **Cybersecurity** standards.

---

## 🏗️ System Architecture

Our end-to-end architecture is structured to prevent data leakage, validate mathematical assumptions, and protect data privacy:

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

## 🛠️ Core Upgrades

### 1. Leak-Proof ML Pipeline
* Encapsulated categorical filling, derived feature engineering (`BuiltRemodelAge`, `NeworOldGarage`), and standardized scaling inside a scikit-learn `Pipeline` and custom `ColumnTransformer` classes.
* Statistics (like medians and scale metrics) are calculated *exclusively* on the training set during training, ensuring **zero data leakage** into test folds or inference rows.
* Utilizes `TransformedTargetRegressor` to automatically manage log-transformation of `SalePrice`.

### 2. Model Risk Management (MRM) Diagnostics
To comply with model validation guidelines (such as SR 11-7), the application automatically audits OLS regression assumptions:
* **Multicollinearity**: Computes Variance Inflation Factors (VIF) on active features.
* **Autocorrelation**: Computes the Durbin-Watson statistic (achieved **2.03**, representing independent residuals).
* **Heteroscedasticity**: Performs the Breusch-Pagan test to ensure constant error variance.
* **Normality of Errors**: Performs the Jarque-Bera test on residuals.
* **Stability Audit**: Continuous monitoring of train R2 (92.1%) vs test R2 (93.4%) gaps.

### 3. Local-First AI Privacy & Cybersecurity
* **Data Isolation**: Integrates local LLM execution via **Ollama** using `llama3.1:8b`. Raw transaction data is never uploaded to cloud APIs.
* **Injection Safeguards**: Text input fields pass through regex-based prompt injection detection.
* **Output Sanitization**: Responses are filtered to strip HTML script/style tags, blocking Cross-Site Scripting (XSS) in the Streamlit frontend.

---

## 📂 Repository Layout

```
.
├── src/
│   ├── core/
│   │   └── interfaces.py       # Decoupled LLMProvider interface
│   ├── providers/
│   │   └── ollama_provider.py   # Concrete local Ollama provider
│   ├── services/
│   │   ├── explanation_service.py # Prompt structuring & LLM coordination
│   │   └── mrm_service.py      # Statistical diagnostic audits
│   ├── utils/
│   │   ├── preprocessing.py    # Custom scikit-learn transformers
│   │   └── security.py         # Input sanitization and prompt injection checks
│   └── data/                   # (Git-ignored) Serialized model pipeline & metadata
├── notebooks/
│   └── Regression_Analysis.ipynb # Research notebook (modernized with pipeline integration)
├── scripts/
│   └── train_model.py          # Automated model training and evaluation script
├── app.py                      # Interactive Streamlit dashboard
├── pyproject.toml              # Ruff configurations
├── requirements.txt            # Pinned dependencies
├── VERSION                     # Version string
├── CHANGELOG.md                # Semantic release history
├── task.md                     # Live progress tracker
└── README.md                   # This document
```

---

## 🚀 Quick Start

### 1. Prerequisites
* Install [Ollama](https://ollama.com/) and run the Llama 3.1 model:
  ```bash
  ollama run llama3.1:8b
  ```

### 2. Installation
Set up a local virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the Pipeline
To compile the leak-proof pipeline, execute CV grid searches, and serialize the model artifacts:
```bash
python scripts/train_model.py
```

### 4. Run the Streamlit Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```

---

## 🧪 Quality and Linting
To check code compliance, format styling, and import organization, run the Ruff linter:
```bash
ruff check .
```
All code files under `src/` and `app.py` are fully lint-compliant.