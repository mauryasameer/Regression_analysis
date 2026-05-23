# Beyond the Jupyter Notebook: Modernizing ML with Local LLMs, Model Risk Management, and Cybersecurity

Jupyter Notebooks are the playground of data science. They are fantastic for exploratory data analysis (EDA), visualizing distributions, and fitting initial models. However, when transitioning from an experimental notebook to an enterprise-grade application, several hidden risks surface:
* **Data Leakage**: Preprocessing data (like computing medians or scaling features) on the entire dataset before splitting leads to optimistic test metrics that fail in production.
* **Model Risk**: Traditional models are often deployed without validating underlying assumptions (e.g., multicollinearity, homoscedasticity), violating regulatory validation frameworks (like SR 11-7).
* **AI Security and Privacy**: Sending private housing, financial, or user data to external cloud LLM APIs exposes institutions to data leakage and prompt injection vulnerabilities.

In this article, I will walk you through how I took a standard housing valuation notebook comparing Ridge and Lasso regression and upgraded it into an **enterprise-ready, secure, and explainable ML application**.

---

## The Architecture: Production-Grade ML meets local GenAI

Here is the blueprint of our modernized system:

```
                  ┌────────────────────────────────────────┐
                  │          Raw House Parameters          │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    scikit-learn Preprocessing Pipeline  │ (Prevents Data Leakage)
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Target-Transformed Lasso Regressor  │ (Predicts Price in $)
                  └─────────┬────────────────────┬─────────┘
                            │                    │
                            ▼                    ▼
     ┌──────────────────────────────┐    ┌──────────────────────────────┐
     │    SHAP Local Attributions   │    │  Model Risk Governance Audit │ (VIF, Normality,
     └──────────────┬───────────────┘    └──────────────────────────────┘  Heteroscedasticity)
                    │
                    ▼
     ┌──────────────────────────────┐
     │    Local Cybersecurity Gate  │ (Input Sanitization & Injection Defense)
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

## 1. Eliminating Data Leakage with scikit-learn Pipelines

In the original notebook, filling missing values and standardizing inputs was done on the entire dataset. To fix this, I created a modular, leak-proof pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.preprocessing import HouseFeatureDeriver, AligningDummifier, FeatureSelector

# Leak-proof preprocessing and feature scaling
prep_pipeline = Pipeline([
    ('deriver', HouseFeatureDeriver()), # Fills NaNs and derives age parameters
    ('dummifier', AligningDummifier(columns=CAT_COLS)), # Aligns categorical dummies
    ('selector', FeatureSelector(features=SELECTED_FEATURES)), # Selects 55 RFE features
    ('scaler', StandardScaler()) # Standardizes
])
```

By encapsulating these in a pipeline, we fit the parameters *only* on the training set and transform the test set. Additionally, we wrap our Lasso model in a `TransformedTargetRegressor` to handle log-transformations of the target price automatically, keeping our inputs and outputs in actual dollars.

---

## 2. Model Risk Management (MRM) & Statistical Diagnostics

Under frameworks like the Federal Reserve's **SR 11-7**, models must go through a formal validation audit. We implemented statistical checkers to validate ordinary least squares (OLS) regression assumptions:

1. **Multicollinearity (VIF)**: We calculate the Variance Inflation Factor for features. Collinear variables like `TotRmsAbvGrd` and `GarageArea` were removed to keep all VIFs below the critical threshold of 5.
2. **Heteroscedasticity (Breusch-Pagan Test)**: We audit the error residuals to ensure constant variance across predictions.
3. **Autocorrelation (Durbin-Watson Test)**: We verify that errors are independent (achieved a statistic of 2.03, showing zero serial correlation).
4. **Normality (Jarque-Bera Test)**: We check if error residuals follow a normal distribution.

In our Streamlit dashboard, these test results are presented in a dedicated **Model Governance** tab, complete with stability metrics (Train R2 of 92.1% vs Test R2 of 93.4%, confirming high stability).

---

## 3. Cybersecurity & Local-First AI Privacy

Integrating Large Language Models (LLMs) to explain predictions is highly valuable, but uploading raw customer data to a public cloud API is a security hazard. 

To address this, the application runs **Llama 3.1 (8B)** entirely locally via **Ollama**. No data egresses the system. To double-down on security, we implemented a dedicated `llm_security_gate`:
* **Range Validation**: Whitelisting categorical inputs and enforcing numerical bounds to block buffer overflows.
* **System Prompt Hardening**: Surrounding variables with strict XML boundaries (`<positive_value_drivers>`) so user queries cannot bypass instructions.
* **Output Sanitization**: Stripping HTML and JavaScript tags from the LLM response to prevent Cross-Site Scripting (XSS) in the Streamlit frontend.

---

## Conclusion: The Final Application

The result is a clean, modern, interactive web application built with Streamlit. It takes raw housing parameters, runs the inference pipeline, visualizes local feature attributions using a **SHAP Waterfall plot**, and generates a secure local AI valuation summary.

By wrapping research notebooks in robust pipelines, auditing statistical risks, and securing LLM integration, we can turn simple notebook experiments into secure, compliant, and highly explaining enterprise systems.

*The full source code, including the local Streamlit dashboard and testing suites, is available on GitHub in the [Regression_analysis](https://github.com/mauryasameer/Regression_analysis) repository.*
