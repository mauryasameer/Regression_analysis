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
