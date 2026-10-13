"""Forecasting models for one horizon: two baselines, logistic regression and LightGBM.

Every model has `fit(frame, y)` and `predict(frame) -> probability of an alert in the target hour`.
`frame` holds the features of `feature_columns()`, with the horizon-specific ones already renamed
(target_hour, target_weekday, target_active_day_ago, target_active_week_ago).
"""

import numpy as np
import pandas as pd

from airalert.features import UNIT_FEATURES
from airalert.telegram_features import TELEGRAM_FEATURES

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
    """`telegram=True` adds the Telegram threat-post features (Q12)."""

    name = "logistic"

    def __init__(self, telegram: bool = False):
        self.numeric = NUMERIC + (TELEGRAM_FEATURES if telegram else [])

    def fit(self, frame, y):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        encode = ColumnTransformer([
            ("numeric", StandardScaler(), self.numeric),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), CALENDAR + ["unit_id"]),
        ])
        self.model = make_pipeline(encode, LogisticRegression(max_iter=300))
        self.model.fit(frame, y)
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame)[:, 1]


class LightGBM:
    """Gradient-boosted trees.

    `early_stopping_days`: choose the number of trees on the last N days of the *training* rows (trees are added
    until the log loss there stops improving for 100 rounds), then refit on all training rows with that number.
    Validation data never sets it. `use_unit=False` leaves out the raion and oblast identity, so the model can
    only use what is happening around the unit. `telegram=True` adds the Telegram threat-post features (Q12).
    """

    name = "lightgbm"

    def __init__(self, early_stopping_days: int | None = None, use_unit: bool = True, telegram: bool = False, **params):
        self.early_stopping_days, self.use_unit = early_stopping_days, use_unit
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, **params}
        self.columns = feature_columns() if use_unit else [c for c in feature_columns() if c not in CATEGORICAL]
        self.columns += TELEGRAM_FEATURES if telegram else []
        self.categorical = [c for c in CATEGORICAL if c in self.columns]

    def fit(self, frame, y):
        import lightgbm as lgb

        y = np.asarray(y)
        params = dict(self.params)
        if self.early_stopping_days:
            inner = (frame["t"] > frame["t"].max() - pd.Timedelta(days=self.early_stopping_days)).to_numpy()
            probe = lgb.LGBMClassifier(**params)
            probe.fit(frame.loc[~inner, self.columns], y[~inner], categorical_feature=self.categorical,
                      eval_X=frame.loc[inner, self.columns], eval_y=y[inner], eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            params["n_estimators"] = max(int(probe.best_iteration_ or params["n_estimators"]), 1)
        self.n_estimators = params["n_estimators"]
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(frame[self.columns], y, categorical_feature=self.categorical)
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame[self.columns])[:, 1]


class PerUnit:
    """One model per unit, each trained only on that unit's rows (Q13). A unit whose training rows are all one
    class gets that constant; a unit never seen in training gets the overall rate.

    The unit models are trained in parallel threads (`n_jobs`); give the base model one thread each (e.g. LightGBM
    with n_jobs=1), since many threads per small model cost more than they save.
    """

    def __init__(self, base_factory, n_jobs: int = 8):
        self.base_factory, self.n_jobs = base_factory, n_jobs
        self.name = f"{getattr(base_factory, 'name', 'model')}_per_unit"

    def fit(self, frame, y):
        from joblib import Parallel, delayed

        y = np.asarray(y)
        self.overall = float(y.mean())
        groups = list(frame.groupby("unit_id", observed=True).indices.items())

        def fit_one(idx):
            y_unit = y[idx]
            return float(y_unit[0]) if np.all(y_unit == y_unit[0]) else self.base_factory().fit(frame.iloc[idx], y_unit)

        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(fit_one)(idx) for _, idx in groups)
        self.models = {str(unit): model for (unit, _), model in zip(groups, fitted)}
        return self

    def predict(self, frame):
        p = np.full(len(frame), self.overall)
        for unit, idx in frame.groupby("unit_id", observed=True).indices.items():
            model = self.models.get(str(unit))
            if isinstance(model, float):
                p[idx] = model
            elif model is not None:
                p[idx] = model.predict(frame.iloc[idx])
        return p


MODELS = {m.name: m for m in (Persistence, HistoricalRate, Logistic, LightGBM)}
