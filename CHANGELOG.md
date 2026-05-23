# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-05-24

### Fixed
- Pinned ruff to `==0.15.14` in `requirements.txt` and CI to prevent local/CI lint divergence.
- Added Python 3.12 to CI test matrix.
- Consolidated `requirements-dev.txt` into `requirements.txt`; removed redundant file.
- Removed committed raw data files (`train.csv`, `bck.png`) and added them to `.gitignore`.

[1.0.1]: https://github.com/mauryasameer/Regression_analysis/compare/v1.0.0...v1.0.1

## [1.0.0] - 2026-05-24

### Added
- Created production-grade machine learning pipelines using scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage.
- Implemented **Model Risk Management (MRM)** statistical diagnostic suite (VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera tests).
- Integrated local LLM capability using Ollama (`llama3.1:8b`) via the **Provider Abstraction Pattern**.
- Established **Cybersecurity** defenses (input sanitization, prompt injection safeguards, XSS output sanitization).
- Developed a root-level Streamlit web dashboard (`app.py`) featuring real-time prediction, interactive SHAP waterfall explanations, and an MRM governance summary.
- Moved research notebook into `notebooks/` directory.
