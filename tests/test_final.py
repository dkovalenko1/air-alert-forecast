"""The test run takes its configuration from the validation decisions only."""

import pandas as pd
import pytest

from airalert.final import configuration
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
    assert list(factories) == ["persistence", "historical_rate", "logistic", "logistic + Telegram (targeted)",
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
