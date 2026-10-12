"""Forecasting models for one horizon: two baselines, logistic regression and LightGBM.

Every model has `fit(frame, y)` and `predict(frame) -> probability of an alert in the target hour`.
`frame` holds the features of `feature_columns()`, with the horizon-specific ones already renamed
(target_hour, target_weekday, target_active_day_ago, target_active_week_ago).
"""

import numpy as np
import pandas as pd

from airalert.features import UNIT_FEATURES

NUMERIC = UNIT_FEATURES + [
    "national_on_alert_share", "national_starts_last_1h", "oblast_on_alert_share",
    "neighbour_on_alert_share", "neighbour_starts_last_1h",
    "target_active_day_ago", "target_active_week_ago",
]
CALENDAR = ["target_hour", "target_weekday", "month"]
CATEGORICAL = ["unit_id", "oblast"]


def feature_columns() -> list[str]:
    return NUMERIC + CALENDAR + CATEGORICAL


class Persistence:
    """'Same as now': alert in the target hour if on alert at t."""

    name = "persistence"

    def fit(self, frame, y):
        return self

    def predict(self, frame):
        return frame["on_alert"].to_numpy(dtype=float)


class HistoricalRate:
    """Share of the last 4 weeks of training hours with an alert, per unit, target hour and weekday."""

    name = "historical_rate"

    def __init__(self, window=pd.Timedelta(days=28)):
        self.window = window

    def fit(self, frame, y):
        recent = frame["t"] >= frame["t"].max() - self.window
        data = frame.loc[recent, ["unit_id", "target_hour", "target_weekday"]].assign(y=np.asarray(y)[recent.to_numpy()])
        self.by_slot = data.groupby(["unit_id", "target_hour", "target_weekday"], observed=True)["y"].mean()
        self.by_unit = data.groupby("unit_id", observed=True)["y"].mean()
        self.overall = data["y"].mean()
        return self

    def predict(self, frame):
        keys = pd.MultiIndex.from_frame(frame[["unit_id", "target_hour", "target_weekday"]].astype(object))
        p = self.by_slot.reindex(keys).to_numpy()
        p = np.where(np.isnan(p), self.by_unit.reindex(frame["unit_id"].astype(object)).to_numpy(), p)
        return np.where(np.isnan(p), self.overall, p)


class Logistic:
    name = "logistic"

    def fit(self, frame, y):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        encode = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), CALENDAR + ["unit_id"]),
        ])
        self.model = make_pipeline(encode, LogisticRegression(max_iter=300))
        self.model.fit(frame, y)
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame)[:, 1]


class LightGBM:
    name = "lightgbm"

    def __init__(self, **params):
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, **params}

    def fit(self, frame, y):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(frame[feature_columns()], y, categorical_feature=CATEGORICAL)
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame[feature_columns()])[:, 1]


MODELS = {m.name: m for m in (Persistence, HistoricalRate, Logistic, LightGBM)}
