import numpy as np
import pandas as pd
import pytest

from airalert.models import CATEGORICAL, NUMERIC, LightGBM, Logistic, PerUnit, RecentRate


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


def test_early_stopping_chooses_trees_on_training_rows_only():
    data, y = frame()
    model = LightGBM(early_stopping_days=7, n_estimators=2000, learning_rate=0.1).fit(data, y)
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
