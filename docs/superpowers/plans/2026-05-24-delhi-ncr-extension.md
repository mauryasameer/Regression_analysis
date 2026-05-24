# Delhi NCR Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Real Estate ML Portal with a Delhi NCR tab supporting flat sale price + rent prediction for Gurgaon, Noida, and Delhi, with a monthly automated data refresh via GitHub Actions.

**Architecture:** Parallel pipeline alongside the existing Ames Iowa model — no changes to existing Ames code except wrapping its UI in a `st.tabs()` container. Per-region, per-mode LassoCV models trained from a Kaggle base CSV and refreshed monthly. Layout C: sidebar region list + Folium map + prediction form with Buy/Rent toggle.

**Tech Stack:** scikit-learn, pandas, numpy, shap, joblib, folium, streamlit-folium, httpx, beautifulsoup4, PyYAML, pytest, pytest-mock

---

## File Map

**New files:**
- `configs/delhi_ncr_regions.yaml` — region config driving sidebar, map, and model loading
- `src/utils/delhi_preprocessing.py` — `DelhiFeatureTransformer`, `DelhiColumnFinalizer`
- `src/services/delhi_mrm_service.py` — thin wrapper around existing MRM functions
- `scripts/train_delhi_ncr.py` — trains sale + rent LassoCV per ready region
- `scripts/scrape_delhi_ncr.py` — scrapes 99acres listings per region
- `models/delhi_ncr/.gitkeep` — tracked empty dir for committed artifacts
- `.github/workflows/delhi_data_refresh.yml` — monthly cron
- `tests/unit/test_delhi_preprocessing.py`
- `tests/unit/test_scraper.py`
- `tests/integration/test_delhi_pipeline.py`
- `tests/test_data/delhi_fixture.csv`

**Modified files:**
- `app.py` — add `st.tabs()` wrapper; add Delhi NCR tab function
- `requirements.txt` — add folium, streamlit-folium, httpx, beautifulsoup4, pyyaml
- `.gitignore` — add `src/data/delhi_ncr/`

---

## Task 1: Scaffold — Dirs, Config, Requirements, Gitignore

**Files:**
- Create: `configs/delhi_ncr_regions.yaml`
- Create: `models/delhi_ncr/.gitkeep`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p configs models/delhi_ncr src/data/delhi_ncr
touch models/delhi_ncr/.gitkeep
```

- [ ] **Step 2: Write `configs/delhi_ncr_regions.yaml`**

```yaml
regions:
  - name: Gurgaon
    lat: 28.4595
    lng: 77.0266
    model_ready: true
    localities:
      - DLF Phase 1
      - DLF Phase 2
      - Sector 56
      - Golf Course Road
      - Sohna Road
  - name: Noida
    lat: 28.5355
    lng: 77.3910
    model_ready: true
    localities:
      - Sector 18
      - Sector 62
      - Sector 137
      - Sector 150
      - Greater Noida
  - name: Delhi
    lat: 28.6139
    lng: 77.2090
    model_ready: true
    localities:
      - Dwarka
      - Rohini
      - Saket
      - Vasant Kunj
      - Lajpat Nagar
      - Janakpuri
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

- [ ] **Step 3: Add new deps to `requirements.txt`**

Add these lines to `requirements.txt`:

```
folium>=0.17.0
streamlit-folium>=0.22.0
httpx>=0.27.0
beautifulsoup4>=4.12.0
pyyaml>=6.0.0
```

- [ ] **Step 4: Update `.gitignore`**

Add this block to `.gitignore`:

```
# Delhi NCR training data — never committed
src/data/delhi_ncr/
```

- [ ] **Step 5: Install new deps**

```bash
venv/bin/pip install folium streamlit-folium httpx beautifulsoup4 pyyaml
```

Expected: `Successfully installed folium-... streamlit-folium-...`

- [ ] **Step 6: Commit**

```bash
git add configs/delhi_ncr_regions.yaml models/delhi_ncr/.gitkeep requirements.txt .gitignore
git commit -m "chore: scaffold Delhi NCR dirs, config, and deps"
```

---

## Task 2: Test Fixture + Failing Preprocessing Tests

**Files:**
- Create: `tests/test_data/delhi_fixture.csv`
- Create: `tests/unit/test_delhi_preprocessing.py`

- [ ] **Step 1: Write `tests/test_data/delhi_fixture.csv`**

```csv
region,bhk,area_sqft,floor,total_floors,age_years,furnishing,locality,parking,lift,metro_dist_km,price_inr,rent_inr
Gurgaon,2,1100,5,15,3,Furnished,DLF Phase 1,1,1,0.8,9200000,35000
Gurgaon,3,1500,8,20,1,Semi-Furnished,Golf Course Road,1,1,1.2,14500000,55000
Noida,2,950,3,10,5,Unfurnished,Sector 62,0,1,2.1,6800000,22000
Noida,3,1650,12,25,2,Furnished,Sector 137,1,1,0.5,13200000,48000
Delhi,2,1000,2,6,8,Semi-Furnished,Dwarka,0,1,1.5,8500000,28000
```

- [ ] **Step 2: Write failing tests in `tests/unit/test_delhi_preprocessing.py`**

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "bhk": 2, "area_sqft": 1100.0, "floor": 5, "total_floors": 15,
            "age_years": 3.0, "furnishing": "Furnished", "locality": "DLF Phase 1",
            "parking": 1, "lift": 1, "metro_dist_km": 0.8,
        },
        {
            "bhk": 3, "area_sqft": None, "floor": 8, "total_floors": 20,
            "age_years": 1.0, "furnishing": "Semi-Furnished", "locality": "Golf Course Road",
            "parking": 1, "lift": 1, "metro_dist_km": 1.2,
        },
    ])


def test_transformer_fills_missing_area(sample_df):
    from src.utils.delhi_preprocessing import DelhiFeatureTransformer
    t = DelhiFeatureTransformer()
    t.fit(sample_df)
    out = t.transform(sample_df)
    assert out["area_sqft"].isna().sum() == 0


def test_transformer_casts_types(sample_df):
    from src.utils.delhi_preprocessing import DelhiFeatureTransformer
    t = DelhiFeatureTransformer()
    t.fit(sample_df)
    out = t.transform(sample_df)
    assert out["bhk"].dtype == np.int64 or out["bhk"].dtype == int
    assert out["parking"].dtype == np.int64 or out["parking"].dtype == int
    assert out["lift"].dtype == np.int64 or out["lift"].dtype == int


def test_column_finalizer_stores_columns(sample_df):
    from src.utils.delhi_preprocessing import DelhiColumnFinalizer
    from src.utils.preprocessing import AligningDummifier

    dummifier = AligningDummifier(columns=["furnishing", "locality"])
    df_dummy = dummifier.fit_transform(sample_df)

    finalizer = DelhiColumnFinalizer()
    finalizer.fit(df_dummy)
    assert len(finalizer.columns_) > 0
    assert "bhk" in finalizer.columns_
    assert "furnishing" not in finalizer.columns_


def test_column_finalizer_pads_missing_column(sample_df):
    from src.utils.delhi_preprocessing import DelhiColumnFinalizer
    from src.utils.preprocessing import AligningDummifier

    dummifier = AligningDummifier(columns=["furnishing", "locality"])
    df_dummy = dummifier.fit_transform(sample_df)

    finalizer = DelhiColumnFinalizer()
    finalizer.fit(df_dummy)

    df_missing = df_dummy.drop(columns=[df_dummy.columns[-1]])
    out = finalizer.transform(df_missing)
    assert list(out.columns) == finalizer.columns_


def test_full_prep_pipeline_returns_array(sample_df):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.utils.delhi_preprocessing import DelhiColumnFinalizer, DelhiFeatureTransformer
    from src.utils.preprocessing import AligningDummifier

    pipe = Pipeline([
        ("transformer", DelhiFeatureTransformer()),
        ("dummifier", AligningDummifier(columns=["furnishing", "locality"])),
        ("finalizer", DelhiColumnFinalizer()),
        ("scaler", StandardScaler()),
    ])
    result = pipe.fit_transform(sample_df)
    assert result.shape[0] == 2
    assert result.shape[1] > 8
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
venv/bin/pytest tests/unit/test_delhi_preprocessing.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.utils.delhi_preprocessing'`

- [ ] **Step 4: Commit**

```bash
git add tests/test_data/delhi_fixture.csv tests/unit/test_delhi_preprocessing.py
git commit -m "test: add Delhi NCR preprocessing tests and fixture data"
```

---

## Task 3: Delhi Preprocessing Module

**Files:**
- Create: `src/utils/delhi_preprocessing.py`

- [ ] **Step 1: Write `src/utils/delhi_preprocessing.py`**

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = [
    "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "metro_dist_km", "parking", "lift",
]
CATEGORICAL_FEATURES = ["furnishing", "locality"]
INT_FEATURES = ["bhk", "floor", "total_floors", "parking", "lift"]


class DelhiFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                self.medians_[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                fill = self.medians_.get(col, 0.0)
                X[col] = X[col].fillna(fill if not pd.isna(fill) else 0.0)
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown")
        for col in INT_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X


class DelhiColumnFinalizer(BaseEstimator, TransformerMixin):
    """Stores the fitted column list and ensures consistent column order at inference."""

    def __init__(self):
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            if col not in X.columns:
                X[col] = 0
        return X[self.columns_]
```

- [ ] **Step 2: Run tests — verify they pass**

```bash
venv/bin/pytest tests/unit/test_delhi_preprocessing.py -v
```

Expected: `5 passed`

- [ ] **Step 3: Commit**

```bash
git add src/utils/delhi_preprocessing.py
git commit -m "feat: add Delhi NCR feature preprocessing transformers"
```

---

## Task 4: Training Script + Integration Test

**Files:**
- Create: `scripts/train_delhi_ncr.py`
- Create: `tests/integration/test_delhi_pipeline.py`

- [ ] **Step 1: Write `tests/integration/test_delhi_pipeline.py`**

This test is skipped in CI if artifacts are absent (matching the existing integration test pattern).

```python
import os

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fixture_df():
    path = "tests/test_data/delhi_fixture.csv"
    return pd.read_csv(path)


def test_gurgaon_sale_pipeline_predicts(fixture_df):
    model_path = "models/delhi_ncr/model_Gurgaon_sale.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Gurgaon sale model artifact not found — run train_delhi_ncr.py first")

    pipeline = joblib.load(model_path)
    row = fixture_df[fixture_df["region"] == "Gurgaon"].drop(
        columns=["region", "price_inr", "rent_inr"]
    ).iloc[:1]
    pred = pipeline.predict(row)
    assert pred.shape == (1,)
    assert pred[0] > 0


def test_gurgaon_rent_pipeline_predicts(fixture_df):
    model_path = "models/delhi_ncr/model_Gurgaon_rent.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Gurgaon rent model artifact not found — run train_delhi_ncr.py first")

    pipeline = joblib.load(model_path)
    row = fixture_df[fixture_df["region"] == "Gurgaon"].drop(
        columns=["region", "price_inr", "rent_inr"]
    ).iloc[:1]
    pred = pipeline.predict(row)
    assert pred.shape == (1,)
    assert pred[0] > 0


def test_metadata_has_expected_keys():
    meta_path = "models/delhi_ncr/metadata_Gurgaon_sale.joblib"
    if not os.path.exists(meta_path):
        pytest.skip("Gurgaon sale metadata not found")
    meta = joblib.load(meta_path)
    for key in ("train_r2", "test_r2", "train_mse", "test_mse", "features"):
        assert key in meta, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests — verify they skip (not fail)**

```bash
venv/bin/pytest tests/integration/test_delhi_pipeline.py -v
```

Expected: `3 skipped`

- [ ] **Step 3: Write `scripts/train_delhi_ncr.py`**

```python
import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.delhi_preprocessing import DelhiColumnFinalizer, DelhiFeatureTransformer
from src.utils.preprocessing import AligningDummifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
FEATURE_COLS = [
    "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "furnishing", "locality",
    "parking", "lift", "metro_dist_km",
]


def build_prep_pipeline() -> Pipeline:
    return Pipeline([
        ("transformer", DelhiFeatureTransformer()),
        ("dummifier", AligningDummifier(columns=["furnishing", "locality"])),
        ("finalizer", DelhiColumnFinalizer()),
        ("scaler", StandardScaler()),
    ])


def train_region_mode(df: pd.DataFrame, region: str, mode: str, target_col: str) -> None:
    logger.info("Training %s | %s", region, mode)
    df_region = df[df["region"] == region].copy()

    if len(df_region) < 30:
        logger.warning("Skipping %s %s — only %d rows", region, mode, len(df_region))
        return

    X = df_region[FEATURE_COLS]
    y = df_region[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=42
    )

    prep_pipeline = build_prep_pipeline()
    X_train_prep = prep_pipeline.fit_transform(X_train)
    X_test_prep = prep_pipeline.transform(X_test)

    regressor = TransformedTargetRegressor(
        regressor=LassoCV(alphas=ALPHAS, cv=5, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    regressor.fit(X_train_prep, y_train)

    y_pred_train = regressor.predict(X_train_prep)
    y_pred_test = regressor.predict(X_test_prep)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    logger.info("  Train R2: %.4f | Test R2: %.4f", train_r2, test_r2)

    fitted_lasso = regressor.regressor_
    explainer = shap.LinearExplainer(fitted_lasso, X_train_prep)

    full_pipeline = Pipeline([("prep", prep_pipeline), ("model", regressor)])
    feature_names = prep_pipeline.named_steps["finalizer"].columns_

    os.makedirs("models/delhi_ncr", exist_ok=True)
    tag = f"{region}_{mode}"
    joblib.dump(full_pipeline, f"models/delhi_ncr/model_{tag}.joblib")
    joblib.dump(explainer, f"models/delhi_ncr/shap_{tag}.joblib")
    joblib.dump(
        {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "features": feature_names,
            "X_train_prep": X_train_prep,
            "X_test_prep": X_test_prep,
            "y_train": y_train.values,
            "y_test": y_test.values,
        },
        f"models/delhi_ncr/metadata_{tag}.joblib",
    )
    logger.info("  Saved artifacts for %s", tag)


def main(only_updated: bool = False) -> None:
    data_path = "src/data/delhi_ncr/base.csv"
    if not os.path.exists(data_path):
        logger.error(
            "Dataset not found at %s. Download from Kaggle and save there.", data_path
        )
        return

    with open("configs/delhi_ncr_regions.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(data_path, encoding="utf-8")
    logger.info("Loaded %d rows from %s", len(df), data_path)

    for region_cfg in config["regions"]:
        if not region_cfg["model_ready"]:
            continue
        region = region_cfg["name"]

        if only_updated:
            live_path = f"src/data/delhi_ncr/{region.lower()}_live.csv"
            if not os.path.exists(live_path):
                continue
            live_df = pd.read_csv(live_path)
            if len(live_df) < 50:
                logger.info("Skipping %s — fewer than 50 new rows", region)
                continue
            df_region_extra = pd.read_csv(live_path)
            df = pd.concat([df, df_region_extra], ignore_index=True)

        train_region_mode(df, region, "sale", "price_inr")
        train_region_mode(df, region, "rent", "rent_inr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-updated", action="store_true")
    args = parser.parse_args()
    main(only_updated=args.only_updated)
```

- [ ] **Step 4: Run integration tests — verify they still skip (not error)**

```bash
venv/bin/pytest tests/integration/test_delhi_pipeline.py -v
```

Expected: `3 skipped`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_delhi_ncr.py tests/integration/test_delhi_pipeline.py
git commit -m "feat: add Delhi NCR training script and integration tests"
```

---

## Task 5: Delhi MRM Service

**Files:**
- Create: `src/services/delhi_mrm_service.py`

- [ ] **Step 1: Write `src/services/delhi_mrm_service.py`**

This is a thin re-export so `app.py` can import from a single Delhi-specific module without coupling to the Ames MRM module directly.

```python
from src.services.mrm_service import audit_model_stability, calculate_vif, run_residual_diagnostics

__all__ = ["audit_model_stability", "calculate_vif", "run_residual_diagnostics"]
```

- [ ] **Step 2: Verify import works**

```bash
venv/bin/python -c "from src.services.delhi_mrm_service import audit_model_stability; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/services/delhi_mrm_service.py
git commit -m "feat: add Delhi MRM service module"
```

---

## Task 6: App Tab Switcher

**Files:**
- Modify: `app.py`

The goal is to wrap the existing Ames Iowa UI content in a `with tab_ames:` block and add an empty `with tab_ncr:` placeholder. No existing Ames logic changes.

- [ ] **Step 1: Find the line in `app.py` where the main UI begins**

```bash
grep -n "st.markdown\|st.header\|st.title\|main-header" app.py | head -10
```

Note the line number where the main header markdown begins (after the CSS block and asset loading).

- [ ] **Step 2: Add tab switcher and wrap Ames content**

In `app.py`, after line 117 (`explanation_service, llm_status = init_llm()`), add:

```python
# ── Tab switcher ──────────────────────────────────────────────────────────────
tab_ames, tab_ncr = st.tabs(["🇺🇸 Ames, Iowa", "🇮🇳 Delhi NCR"])
```

Then wrap all remaining UI content (from the first `st.markdown("""<div...` header down to end of file) inside `with tab_ames:` by indenting it. Add at the end of the file:

```python
with tab_ncr:
    st.info("Delhi NCR tab — coming in next task.")
```

- [ ] **Step 3: Run app and verify both tabs appear**

```bash
venv/bin/streamlit run app.py --server.headless true &
sleep 5
curl -s http://localhost:8501/_stcore/health
kill %1
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit tab switcher for Ames Iowa and Delhi NCR"
```

---

## Task 7: Delhi NCR Tab UI

**Files:**
- Modify: `app.py`

Replaces the `st.info("Delhi NCR tab — coming in next task.")` placeholder with the full Layout C UI.

- [ ] **Step 1: Add new imports to top of `app.py`**

Add these lines alongside the existing imports at the top of `app.py` (after `import shap`):

```python
import folium
import yaml
from streamlit_folium import st_folium
```

- [ ] **Step 2: Add `load_delhi_assets()` function to `app.py`**

Add this function after the existing `load_assets()` function (around line 87):

```python
@st.cache_resource
def load_delhi_assets(region: str, mode: str):
    base = f"models/delhi_ncr"
    tag = f"{region}_{mode}"
    try:
        pipeline = joblib.load(f"{base}/model_{tag}.joblib")
        metadata = joblib.load(f"{base}/metadata_{tag}.joblib")
        explainer = joblib.load(f"{base}/shap_{tag}.joblib")
        return pipeline, metadata, explainer, True
    except Exception as e:
        return None, None, None, False
```

- [ ] **Step 2: Add `load_delhi_regions()` function**

```python
@st.cache_data
def load_delhi_regions():
    import yaml
    with open("configs/delhi_ncr_regions.yaml") as f:
        return yaml.safe_load(f)["regions"]
```

- [ ] **Step 3: Replace the placeholder in `with tab_ncr:` with the full UI**

Replace `st.info("Delhi NCR tab — coming in next task.")` with:

```python
    regions = load_delhi_regions()

    st.markdown('<p class="sub-header">Delhi NCR Flat Price Predictor — Sale & Rent</p>', unsafe_allow_html=True)

    ncr_col1, ncr_col2, ncr_col3 = st.columns([1, 3, 2])

    # ── Sidebar: Region list ──────────────────────────────────────────────────
    with ncr_col1:
        st.markdown("**Select Region**")
        ready_regions = [r for r in regions if r["model_ready"]]
        locked_regions = [r for r in regions if not r["model_ready"]]

        region_names = [r["name"] for r in ready_regions]
        selected_region_name = st.radio(
            "Available models",
            options=region_names,
            label_visibility="collapsed",
        )
        if locked_regions:
            st.markdown("**Coming soon**")
            for r in locked_regions:
                st.markdown(f"<span style='color:#475569'>⚫ {r['name']}</span>", unsafe_allow_html=True)

    selected_region = next(r for r in regions if r["name"] == selected_region_name)

    # ── Centre: Folium map ────────────────────────────────────────────────────
    with ncr_col2:
        m = folium.Map(location=[28.55, 77.25], zoom_start=10, tiles="CartoDB dark_matter")
        for r in regions:
            color = "green" if r["model_ready"] else "gray"
            tooltip = r["name"] + (" ✓" if r["model_ready"] else " — coming soon")
            folium.CircleMarker(
                location=[r["lat"], r["lng"]],
                radius=10 if r["model_ready"] else 7,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9 if r["model_ready"] else 0.4,
                tooltip=tooltip,
                popup=folium.Popup(
                    f"<b>{r['name']}</b><br>{'Model ready ✓' if r['model_ready'] else 'Coming soon'}",
                    max_width=150,
                ),
            ).add_to(m)

        map_result = st_folium(m, height=420, width="100%", returned_objects=["last_object_clicked_tooltip"])

        if map_result and map_result.get("last_object_clicked_tooltip"):
            clicked_name = map_result["last_object_clicked_tooltip"].replace(" ✓", "").strip()
            clicked_region = next((r for r in ready_regions if r["name"] == clicked_name), None)
            if clicked_region:
                selected_region_name = clicked_region["name"]
                selected_region = clicked_region

    # ── Right: Prediction form ────────────────────────────────────────────────
    with ncr_col3:
        st.markdown(f"**Region: 🟢 {selected_region_name}**")

        mode = st.radio("Mode", ["Buy (Sale Price)", "Rent (Monthly)"], horizontal=True)
        mode_key = "sale" if "Buy" in mode else "rent"

        localities = selected_region.get("localities", [])
        locality = st.selectbox("Locality", localities)
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
        area_sqft = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1200, step=50)
        floor = st.number_input("Floor", min_value=0, max_value=60, value=5)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=60, value=15)
        age_years = st.number_input("Property Age (years)", min_value=0, max_value=50, value=3)
        furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
        col_p, col_l = st.columns(2)
        with col_p:
            parking = st.checkbox("Parking", value=True)
        with col_l:
            lift = st.checkbox("Lift", value=True)
        metro_dist_km = st.number_input("Metro Distance (km)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

        if st.button("Predict Price →", use_container_width=True):
            pipeline, metadata, explainer, loaded = load_delhi_assets(selected_region_name, mode_key)
            if not loaded:
                st.error(
                    f"No model found for {selected_region_name} ({mode_key}). "
                    "Run `python scripts/train_delhi_ncr.py` after downloading the dataset."
                )
            else:
                input_df = pd.DataFrame([{
                    "bhk": bhk,
                    "area_sqft": float(area_sqft),
                    "floor": int(floor),
                    "total_floors": int(total_floors),
                    "age_years": float(age_years),
                    "furnishing": furnishing,
                    "locality": locality,
                    "parking": int(parking),
                    "lift": int(lift),
                    "metro_dist_km": float(metro_dist_km),
                }])

                prediction = pipeline.predict(input_df)[0]
                label = "Estimated Sale Price" if mode_key == "sale" else "Estimated Monthly Rent"
                unit = "₹" 
                if prediction >= 10_000_000:
                    display = f"{unit}{prediction/10_000_000:.2f} Cr"
                elif prediction >= 100_000:
                    display = f"{unit}{prediction/100_000:.2f} L"
                else:
                    display = f"{unit}{prediction:,.0f}"

                st.success(f"**{label}:** {display}")

                # SHAP waterfall
                prep_pipeline = pipeline.named_steps["prep"]
                X_prep = prep_pipeline.transform(input_df)
                shap_values = explainer.shap_values(X_prep)
                feature_names = metadata["features"]

                exp_plot = shap.Explanation(
                    values=np.array(shap_values[0]),
                    base_values=explainer.expected_value,
                    data=X_prep[0],
                    feature_names=feature_names,
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                shap.plots.bar(exp_plot, max_display=8, show=False)
                plt.title("Feature Attribution (SHAP)", fontsize=10, pad=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # MRM diagnostics
                with st.expander("Model Risk Diagnostics"):
                    from src.services.delhi_mrm_service import audit_model_stability
                    stability = audit_model_stability(
                        metadata["train_r2"], metadata["test_r2"],
                        metadata["train_mse"], metadata["test_mse"],
                    )
                    st.json(stability)
```

- [ ] **Step 4: Verify app starts without errors**

```bash
venv/bin/streamlit run app.py --server.headless true &
sleep 6
curl -s http://localhost:8501/_stcore/health
kill %1
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add Delhi NCR tab with Layout C map UI and Buy/Rent prediction"
```

---

## Task 8: Scraper + Tests

**Files:**
- Create: `scripts/scrape_delhi_ncr.py`
- Create: `tests/unit/test_scraper.py`

- [ ] **Step 1: Write failing scraper schema test in `tests/unit/test_scraper.py`**

```python
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


EXPECTED_COLUMNS = {
    "region", "bhk", "area_sqft", "floor", "total_floors",
    "age_years", "furnishing", "locality", "parking", "lift",
    "metro_dist_km", "price_inr", "rent_inr",
}

MOCK_HTML = """
<html><body>
<div class="_3bgq4">
  <div class="_2ipzj">3 BHK Flat</div>
  <div class="_1hziv">&#x20B9; 1.2 Cr</div>
  <div class="_2r7aN">1450 sqft</div>
  <div class="_3Cp4n">DLF Phase 1, Gurgaon</div>
  <div class="d-flex">Floor 8 of 15</div>
</div>
</body></html>
"""


def test_parse_listings_returns_dataframe():
    from scripts.scrape_delhi_ncr import parse_listings_html
    result = parse_listings_html(MOCK_HTML, region="Gurgaon")
    assert isinstance(result, pd.DataFrame)


def test_parse_listings_has_required_columns():
    from scripts.scrape_delhi_ncr import parse_listings_html
    result = parse_listings_html(MOCK_HTML, region="Gurgaon")
    for col in EXPECTED_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_deduplicate_removes_duplicate_rows():
    from scripts.scrape_delhi_ncr import deduplicate
    df = pd.DataFrame([
        {"locality": "DLF Phase 1", "area_sqft": 1100, "floor": 5, "price_inr": 9200000},
        {"locality": "DLF Phase 1", "area_sqft": 1100, "floor": 5, "price_inr": 9200000},
        {"locality": "Sector 62",   "area_sqft": 950,  "floor": 3, "price_inr": 6800000},
    ])
    result = deduplicate(df)
    assert len(result) == 2


def test_fetch_does_not_make_real_http_calls():
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = MOCK_HTML
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        from scripts.scrape_delhi_ncr import fetch_region_listings
        result = fetch_region_listings("Gurgaon", max_pages=1)
        assert isinstance(result, pd.DataFrame)
        mock_client.return_value.__enter__.return_value.get.assert_called()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/bin/pytest tests/unit/test_scraper.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'scripts.scrape_delhi_ncr'`

- [ ] **Step 3: Write `scripts/scrape_delhi_ncr.py`**

```python
import hashlib
import logging
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
import pandas as pd
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}
BASE_URL = "https://www.99acres.com/search/property/buy/{region}?city={city_id}&preference=S&area_unit=1&res_com=R"
CITY_IDS = {"Gurgaon": 12, "Noida": 21, "Delhi": 6, "Faridabad": 35, "Ghaziabad": 36}
REQUEST_DELAY = 2.0


def parse_listings_html(html: str, region: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("div._3bgq4"):
        try:
            title_tag = card.select_one("div._2ipzj")
            price_tag = card.select_one("div._1hziv")
            area_tag = card.select_one("div._2r7aN")
            locality_tag = card.select_one("div._3Cp4n")
            floor_tag = card.select_one("div.d-flex")

            if not all([title_tag, price_tag, area_tag]):
                continue

            title = title_tag.get_text(strip=True)
            bhk = int(title.split()[0]) if title[0].isdigit() else 2
            price_text = price_tag.get_text(strip=True).replace("₹", "").replace(",", "").strip()
            price_inr = _parse_inr(price_text)
            area_sqft = float(area_tag.get_text(strip=True).replace("sqft", "").strip())
            locality = locality_tag.get_text(strip=True).split(",")[0] if locality_tag else "Unknown"
            floor_text = floor_tag.get_text(strip=True) if floor_tag else "Floor 1 of 5"
            floor, total_floors = _parse_floor(floor_text)

            rows.append({
                "region": region,
                "bhk": bhk,
                "area_sqft": area_sqft,
                "floor": floor,
                "total_floors": total_floors,
                "age_years": 5.0,
                "furnishing": "Semi-Furnished",
                "locality": locality,
                "parking": 1,
                "lift": 1,
                "metro_dist_km": 1.5,
                "price_inr": price_inr,
                "rent_inr": 0,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _parse_inr(text: str) -> float:
    text = text.lower().replace(",", "")
    if "cr" in text:
        return float(text.replace("cr", "").strip()) * 10_000_000
    if "l" in text or "lac" in text:
        return float(text.replace("lac", "").replace("l", "").strip()) * 100_000
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_floor(text: str) -> tuple[int, int]:
    try:
        parts = text.lower().replace("floor", "").replace("of", "/").split("/")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return 1, 5


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    def _hash(row):
        key = f"{row.get('locality','')}|{row.get('area_sqft',0)}|{row.get('floor',0)}|{row.get('price_inr',0)}"
        return hashlib.sha256(key.encode()).hexdigest()
    df = df.copy()
    df["_hash"] = df.apply(_hash, axis=1)
    df = df.drop_duplicates(subset="_hash").drop(columns=["_hash"])
    return df.reset_index(drop=True)


def fetch_region_listings(region: str, max_pages: int = 3) -> pd.DataFrame:
    city_id = CITY_IDS.get(region, 6)
    all_rows = []
    with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
        for page in range(1, max_pages + 1):
            url = BASE_URL.format(region=region.lower(), city_id=city_id) + f"&page={page}"
            logger.info("Fetching %s page %d", region, page)
            try:
                resp = client.get(url)
                resp.raise_for_status()
                df_page = parse_listings_html(resp.text, region)
                all_rows.append(df_page)
            except Exception as e:
                logger.warning("Failed to fetch %s page %d: %s", region, page, e)
            time.sleep(REQUEST_DELAY)
    if not all_rows:
        return pd.DataFrame()
    return deduplicate(pd.concat(all_rows, ignore_index=True))


def main() -> None:
    with open("configs/delhi_ncr_regions.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs("src/data/delhi_ncr", exist_ok=True)

    for region_cfg in config["regions"]:
        if not region_cfg["model_ready"]:
            continue
        region = region_cfg["name"]
        df_new = fetch_region_listings(region, max_pages=5)

        if df_new.empty:
            logger.warning("No listings fetched for %s", region)
            continue

        out_path = f"src/data/delhi_ncr/{region.lower()}_live.csv"
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            df_combined = deduplicate(pd.concat([df_existing, df_new], ignore_index=True))
        else:
            df_combined = df_new

        df_combined.to_csv(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df_combined), out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run scraper tests — verify they pass**

```bash
venv/bin/pytest tests/unit/test_scraper.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/scrape_delhi_ncr.py tests/unit/test_scraper.py
git commit -m "feat: add 99acres scraper and schema tests"
```

---

## Task 9: GitHub Actions Monthly Refresh Workflow

**Files:**
- Create: `.github/workflows/delhi_data_refresh.yml`

- [ ] **Step 1: Write `.github/workflows/delhi_data_refresh.yml`**

```yaml
name: Delhi NCR Monthly Data Refresh

on:
  schedule:
    - cron: "0 20 1 * *"   # 1st of month, 2am IST (8:30pm UTC prev day)
  workflow_dispatch:        # manual trigger

jobs:
  refresh:
    name: Scrape, Retrain, PR
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Scrape new listings
        run: python scripts/scrape_delhi_ncr.py
        continue-on-error: true

      - name: Retrain updated regions
        run: python scripts/train_delhi_ncr.py --only-updated

      - name: Check for changed artifacts
        id: changes
        run: |
          git diff --quiet models/delhi_ncr/ || echo "changed=true" >> $GITHUB_OUTPUT

      - name: Create PR branch and commit artifacts
        if: steps.changes.outputs.changed == 'true'
        run: |
          BRANCH="data/refresh-$(date +%Y-%m)"
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git checkout -b "$BRANCH"
          git add models/delhi_ncr/
          git commit -m "chore: refresh Delhi NCR model artifacts $(date +%Y-%m)"
          git push origin "$BRANCH"
          gh pr create \
            --title "chore: Delhi NCR model refresh $(date +%Y-%m)" \
            --body "Monthly automated model refresh. Scrape + retrain completed. Review and merge to update production models." \
            --base dev \
            --head "$BRANCH"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Open issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Delhi NCR refresh failed — ${new Date().toISOString().slice(0,7)}`,
              body: `The monthly Delhi NCR data refresh workflow failed. Check the [workflow run](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}) for details.`,
              labels: ["bug", "data-pipeline"]
            })
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/delhi_data_refresh.yml
git commit -m "chore: add monthly Delhi NCR data refresh GitHub Actions workflow"
```

---

## Task 10: Full Test Suite + Final Verification

- [ ] **Step 1: Run full test suite**

```bash
venv/bin/pytest tests/ -v --tb=short
```

Expected: all unit tests pass; integration tests skip (no artifacts yet)

- [ ] **Step 2: Run ruff**

```bash
venv/bin/ruff check .
```

Expected: `All checks passed.`

- [ ] **Step 3: Push branch and open PR to dev**

```bash
git push -u origin feature/delhi-ncr-extension
gh pr create \
  --title "feat: Delhi NCR flat price + rent predictor" \
  --body "$(cat <<'EOF'
## Summary
- Adds Delhi NCR tab with Layout C UI (sidebar + Leaflet map + Buy/Rent form)
- Per-region LassoCV models for sale price and rent (Gurgaon, Noida, Delhi)
- 99acres scraper with deduplication
- Monthly GitHub Actions refresh workflow with auto-PR
- Full test coverage: preprocessing, scraper schema, integration

## Test plan
- [ ] Run `python scripts/train_delhi_ncr.py` after downloading Kaggle dataset
- [ ] Open app and switch to Delhi NCR tab
- [ ] Select a region from sidebar and map
- [ ] Toggle Buy/Rent, fill form, predict
- [ ] Verify SHAP waterfall renders
- [ ] Verify MRM diagnostics expand
EOF
)" \
  --base dev
```
