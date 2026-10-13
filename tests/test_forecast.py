"""A forecast never uses a model trained after its time, or data its inputs could not have had."""

import pandas as pd
import pytest

from airalert.forecast import check_forecast_time

CUTOFF, END = pd.Timestamp("2026-10-12 12:00", tz="UTC"), pd.Timestamp("2026-10-12 21:50", tz="UTC")


def test_forecast_at_or_after_the_cutoff_within_the_data_is_allowed():
    check_forecast_time(CUTOFF, CUTOFF, END)
    check_forecast_time(pd.Timestamp("2026-10-12 21:00", tz="UTC"), CUTOFF, END)


@pytest.mark.parametrize("at", ["2026-10-12 11:00",  # before the model's cutoff
                                "2026-10-12 22:00",  # after the data ends
                                "2026-10-12 13:30"])  # not a whole hour
def test_other_forecast_times_are_refused(at):
    with pytest.raises(SystemExit):
        check_forecast_time(pd.Timestamp(at, tz="UTC"), CUTOFF, END)


class Stub:
    """A model whose forecast is readable from the features: 0.5 if the unit is on alert at t, plus h / 100."""

    def __init__(self, h):
        self.h = h

    def predict(self, frame):
        return frame["on_alert"].to_numpy() * 0.5 + self.h / 100


def test_each_forecast_row_belongs_to_its_own_unit_and_horizon():
    from airalert.features import build_grid
    from airalert.forecast import forecast

    at = pd.Timestamp("2026-08-03 12:00", tz="UTC")
    periods = pd.DataFrame({"unit_id": ["A/a"], "started_at": [at - pd.Timedelta(hours=1)], "finished_at": [at + pd.Timedelta(hours=2)]})
    grid = build_grid(periods, at - pd.Timedelta(hours=5), at + pd.Timedelta(hours=1), pd.Timestamp("2026-08-20", tz="UTC"), ("A/a", "A/b", "B/c"))
    table = forecast(at, {"model": "stub", "models": {h: Stub(h) for h in (1, 6, 24)}}, grid).set_index("unit")
    assert set(table.index) == {"A/a", "A/b", "B/c"}  # only the rows at `at`, one per unit
    assert table.loc["A/a", "on alert"] and not table.loc["A/b", "on alert"] and not table.loc["B/c", "on alert"]
    assert table.loc["A/a", ["h=1", "h=6", "h=24"]].tolist() == pytest.approx([0.51, 0.56, 0.74])
    assert table.loc["A/b", ["h=1", "h=6", "h=24"]].tolist() == pytest.approx([0.01, 0.06, 0.24])


def test_predict_uses_the_saved_model_with_the_latest_cutoff(tmp_path):
    import pickle

    from airalert.forecast import latest_saved

    for name, cutoff in (("old", "2026-10-10 00:00"), ("new", "2026-10-12 00:00"), ("middle", "2026-10-11 00:00")):
        with (tmp_path / f"{name}.pkl").open("wb") as f:
            pickle.dump({"model": name, "cutoff": pd.Timestamp(cutoff, tz="UTC"), "models": {}}, f)
    assert latest_saved(tmp_path)["model"] == "new"
    with pytest.raises(SystemExit, match="train"):
        latest_saved(tmp_path / "nothing-saved-here")
