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
