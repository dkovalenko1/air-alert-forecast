import numpy as np
import pandas as pd
import pytest

from airalert.models import CATEGORICAL, NUMERIC, LightGBM, Logistic, PerUnit, RecentRate, RecentRateByState


def frame(n_days=40, seed=0):
    """Two units: A alerts at night, B never. Features carry the signal through the target hour."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-07-01", periods=n_days * 24, freq="h", tz="UTC")
    rows = []
    for unit, oblast in (("A/a", "A"), ("B/b", "B")):
        hour = t.hour.to_numpy()
        y = ((unit == "A/a") & (hour < 6) & (rng.random(len(t)) < 0.9)).astype(int)
        part = pd.DataFrame({c: rng.random(len(t)) for c in NUMERIC})
        part["target_active_day_ago"] = y  # a strong, honest-looking signal
        part = part.assign(t=t, target_hour=hour, target_weekday=t.weekday, month=t.month, unit_id=unit, oblast=oblast, y=y)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out, out.pop("y").to_numpy()


def test_early_stopping_chooses_the_number_of_trees_on_the_last_training_days_only(monkeypatch):
    """The probe model is fitted on the earlier rows and stopped on the last 7 days of the training rows; the final
    model is then fitted on all of them with that number of trees. Nothing outside the training rows is involved."""
    import lightgbm as lgb

    data, y = frame()
    fits = []
    original = lgb.LGBMClassifier.fit

    def spy(self, X, target, *args, **kwargs):
        fits.append((X.index, kwargs.get("eval_X").index if kwargs.get("eval_X") is not None else None))
        return original(self, X, target, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy)
    model = LightGBM(early_stopping_days=7, n_estimators=2000, learning_rate=0.1).fit(data, y)
    (probe_rows, holdout_rows), (final_rows, no_holdout) = fits
    last_week = data.index[data.t > data.t.max() - pd.Timedelta(days=7)]
    assert holdout_rows.equals(last_week) and probe_rows.intersection(holdout_rows).empty
    assert len(probe_rows) + len(holdout_rows) == len(data) and no_holdout is None and final_rows.equals(data.index)
    assert 1 <= model.n_estimators < 2000
    p = model.predict(data)
    assert ((p >= 0) & (p <= 1)).all() and p[y == 1].mean() > p[y == 0].mean() + 0.5


def test_lightgbm_without_unit_identity_ignores_unit_columns():
    data, y = frame()
    model = LightGBM(use_unit=False, n_estimators=20).fit(data, y)
    assert not set(CATEGORICAL) & set(model.columns)
    renamed = data.assign(unit_id=pd.Categorical(["Z/z"] * len(data)), oblast=pd.Categorical(["Z"] * len(data)))
    np.testing.assert_allclose(model.predict(data), model.predict(renamed))


def test_per_unit_trains_one_model_per_unit_and_handles_one_class_units():
    data, y = frame()
    model = PerUnit(Logistic).fit(data, y)
    assert set(model.models) == {"A/a", "B/b"} and model.models["B/b"] == 0.0  # B never alerts: a constant
    p = model.predict(data)
    assert (p[(data.unit_id == "B/b").to_numpy()] == 0).all()
    unseen = data[data.unit_id == "A/a"].assign(unit_id=pd.Categorical(["C/c"] * (data.unit_id == "A/a").sum()))
    np.testing.assert_allclose(model.predict(unseen), y.mean())


@pytest.mark.parametrize("feature_set", ["all", "targeted"])
def test_telegram_option_adds_exactly_that_set_of_columns(feature_set):
    from airalert.telegram_features import ALL_TELEGRAM_FEATURES, TELEGRAM_SETS

    data, y = frame()
    data = data.assign(**{c: np.random.default_rng(1).random(len(data)) for c in ALL_TELEGRAM_FEATURES})
    with_tg, without = LightGBM(telegram=feature_set, n_estimators=10), LightGBM(n_estimators=10)
    assert set(with_tg.columns) - set(without.columns) == set(TELEGRAM_SETS[feature_set])
    assert len(with_tg.fit(data, y).predict(data)) == len(data)
    assert len(Logistic(telegram=feature_set).fit(data, y).predict(data)) == len(data)


def test_recent_rate_is_each_units_share_of_alert_hours_in_the_last_four_weeks():
    data, y = frame(n_days=40)
    model = RecentRate().fit(data, y)
    recent = (data.t >= data.t.max() - pd.Timedelta(days=28)).to_numpy()
    expected_a = y[recent & (data.unit_id == "A/a").to_numpy()].mean()
    p = model.predict(data)
    assert np.allclose(p[(data.unit_id == "A/a").to_numpy()], expected_a) and (p[(data.unit_id == "B/b").to_numpy()] == 0).all()
    unseen = data.iloc[:3].assign(unit_id=pd.Categorical(["C/c"] * 3))
    assert np.allclose(model.predict(unseen), y[recent].mean())


def test_recent_rate_by_state_splits_each_units_share_by_its_state_at_t():
    data, y = frame(n_days=40)
    data["on_alert"] = np.random.default_rng(2).integers(2, size=len(data))
    model = RecentRateByState().fit(data, y)
    recent = (data.t >= data.t.max() - pd.Timedelta(days=28)).to_numpy()
    p = model.predict(data)
    for state in (0, 1):
        rows = (data.unit_id == "A/a").to_numpy() & (data.on_alert == state).to_numpy()
        assert np.allclose(p[rows], y[recent & rows].mean())
    unseen = data.iloc[:2].assign(unit_id=pd.Categorical(["C/c"] * 2), on_alert=[0, 1])
    assert np.allclose(model.predict(unseen), [y[recent & (data.on_alert == s).to_numpy()].mean() for s in (0, 1)])
