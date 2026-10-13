"""The test run takes its configuration from the validation decisions only."""

import numpy as np
import pandas as pd
import pytest

from airalert.final import comparisons, configuration, differences, differences_by_horizon
from airalert.models import LightGBM, Logistic

DECISIONS = {
    "e1_training_start": {"logistic": "2025-11-01", "lightgbm": "2025-11-01"},
    "e2_shared_or_per_unit": {"logistic": "shared", "lightgbm": "shared"},
    "e3_lightgbm_settings": {"lightgbm": "no unit identity", "e2_mode": "shared", "e2_assumed": True},
    "e4_telegram": {"logistic": "without", "lightgbm": "without"},
    "e5_telegram_targeted": {"logistic": "without", "lightgbm": "without"},
}


def test_chosen_configuration_and_the_telegram_ablation():
    start, factories, chosen = configuration(DECISIONS)
    assert start == pd.Timestamp("2025-11-01", tz="Europe/Kyiv")
    assert list(factories) == ["persistence", "historical_rate", "recent_rate", "logistic", "logistic + Telegram (targeted)",
                               "lightgbm", "lightgbm + Telegram (targeted)"]
    assert chosen == {"logistic": "logistic", "lightgbm": "lightgbm"}
    lightgbm = factories["lightgbm"]()
    assert isinstance(lightgbm, LightGBM) and not lightgbm.use_unit and "unit_id" not in lightgbm.columns
    assert "tg_local_threat_60m" in factories["lightgbm + Telegram (targeted)"]().columns
    assert isinstance(factories["logistic"](), Logistic)


def test_a_telegram_set_chosen_on_validation_becomes_the_main_variant():
    decisions = {**DECISIONS, "e5_telegram_targeted": {"logistic": "targeted Telegram", "lightgbm": "without"}}
    _, factories, chosen = configuration(decisions)
    assert chosen["logistic"] == "logistic + Telegram (targeted)"
    assert [k for k in factories if k.startswith("logistic")] == ["logistic + Telegram (targeted)", "logistic"]


def test_different_training_starts_are_refused():
    decisions = {**DECISIONS, "e1_training_start": {"logistic": "2025-08-01", "lightgbm": "2025-11-01"}}
    with pytest.raises(SystemExit):
        configuration(decisions)


def test_comparisons_pair_each_model_with_the_chosen_lightgbm_and_each_telegram_variant_with_its_model():
    _, factories, chosen = configuration(DECISIONS)
    pairs = comparisons(list(factories), chosen)
    assert ("persistence", "lightgbm") in pairs and ("lightgbm + Telegram (targeted)", "lightgbm") in pairs
    assert ("logistic + Telegram (targeted)", "logistic") in pairs and len(pairs) == len(factories)


def test_differences_of_a_perfect_and_a_coin_flip_model():
    t = pd.date_range("2026-09-14", periods=24 * 10, freq="h", tz="Europe/Kyiv")
    y = np.random.default_rng(0).integers(2, size=len(t))
    rows = [pd.DataFrame({"model": m, "horizon": h, "t": t, "y": y, "p": p})
            for m, p in (("perfect", y.astype(float)), ("coin", np.full(len(t), 0.5))) for h in (1, 2)]
    table = differences(pd.concat(rows), [("coin", "perfect"), ("perfect", "perfect")], n=200)
    coin, same = table.iloc[0], table.iloc[1]
    assert coin.difference == pytest.approx(0.25) and coin.low == pytest.approx(0.25) and coin.share_b_better == 1
    assert same.difference == 0 and same.low == same.high == 0


def test_differences_by_horizon_and_for_calm_units():
    t = pd.date_range("2026-09-14", periods=24 * 10, freq="h", tz="Europe/Kyiv")
    y = np.random.default_rng(1).integers(2, size=len(t))
    on_alert = np.arange(len(t)) % 2  # every other row on alert at t
    rows = [pd.DataFrame({"model": "perfect", "horizon": h, "t": t, "y": y, "p": y.astype(float), "on_alert": on_alert}) for h in (1, 24)]
    # "late" is perfect 1 h ahead but a coin flip a day ahead, and only on calm rows
    rows += [pd.DataFrame({"model": "late", "horizon": 1, "t": t, "y": y, "p": y.astype(float), "on_alert": on_alert}),
             pd.DataFrame({"model": "late", "horizon": 24, "t": t, "y": y, "p": np.where(on_alert == 0, 0.5, y), "on_alert": on_alert})]
    table = differences_by_horizon(pd.concat(rows), [("late", "perfect")]).set_index(["subset", "horizon"])
    assert table.loc[("all", 1), "difference"] == 0 and table.loc[("calm at t", 1), "difference"] == 0
    assert table.loc[("calm at t", 24), "difference"] == pytest.approx(0.25)
    assert table.loc[("all", 24), "difference"] == pytest.approx(0.125)
