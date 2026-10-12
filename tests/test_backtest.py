import numpy as np
import pandas as pd

from airalert.backtest import near_permanent_units, run, score, weeks
from airalert.features import HOUR
from airalert.models import MODELS
from airalert.features import build_grid

UNITS = ("A/a", "A/b", "B/c", "B/d")

START = pd.Timestamp("2026-07-01", tz="UTC")
PERIOD = (pd.Timestamp("2026-08-03", tz="UTC"), pd.Timestamp("2026-08-17", tz="UTC"))


def synthetic_periods(seed=0) -> pd.DataFrame:
    """Units with a daily rhythm: alerts tend to start around 02:00, so time of day is predictable."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in ("A/a", "A/b", "B/c", "B/d"):
        for day in pd.date_range("2026-06-20", "2026-08-25", freq="D", tz="UTC"):
            if rng.random() < 0.8:
                start = day + pd.Timedelta(hours=2, minutes=int(rng.integers(0, 60)))
                rows.append((unit, start, start + pd.Timedelta(minutes=int(rng.integers(20, 180)))))
    rows.append(("B/d", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC")))  # always on alert
    return pd.DataFrame(rows, columns=["unit_id", "started_at", "finished_at"])


def test_walk_forward_runs_every_model_and_beats_nothing_silly():
    grid = build_grid(synthetic_periods(), START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"), UNITS)
    predictions = run(grid, PERIOD, ["persistence", "historical_rate", "logistic", "lightgbm"], horizons=(1, 6), train_start=START)
    assert set(predictions.model) == {"persistence", "historical_rate", "logistic", "lightgbm"}
    assert predictions.p.between(0, 1).all()
    assert predictions.week.nunique() == 2 and predictions.t.min() >= PERIOD[0] and predictions.t.max() < PERIOD[1]
    metrics = score(predictions).set_index(["model", "horizon", "subset"])
    # At 6 h ahead, the daily rhythm makes the historical rate far better than "same as now".
    assert metrics.loc[("historical_rate", 6, "all"), "brier"] < metrics.loc[("persistence", 6, "all"), "brier"]


def test_near_permanent_units_and_excluding_them():
    grid = build_grid(synthetic_periods(), START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"), UNITS)
    assert near_permanent_units(grid, *PERIOD) == {"B/d"}
    predictions = run(grid, PERIOD, ["persistence"], horizons=(1,), train_start=START)
    assert score(predictions, exclude_units={"B/d"}).n.sum() < score(predictions).n.sum()


def labels_cut_at(periods: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The same history up to `cutoff`, and no alert at all after it: every outcome after the cutoff changes."""
    kept = periods[periods.started_at < cutoff].copy()
    kept["finished_at"] = kept.finished_at.clip(upper=cutoff)
    return kept


def test_training_rows_end_by_the_cutoff(monkeypatch):
    fits = []

    class Spy(MODELS["historical_rate"]):
        def fit(self, frame, y):
            fits.append(frame.t.max())
            return super().fit(frame, y)

    monkeypatch.setitem(MODELS, "spy", Spy)
    grid = build_grid(synthetic_periods(), START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"), UNITS)
    for h in (1, 6, 24):
        fits.clear()
        run(grid, PERIOD, ["spy"], horizons=(h,), train_start=START)
        cutoffs = [week_start for week_start, _ in weeks(*PERIOD)]
        # the newest training row's label hour ends exactly at the cutoff: nothing later, nothing needlessly dropped
        assert [t + h * HOUR for t in fits] == cutoffs


def test_outcomes_after_the_cutoff_cannot_change_that_weeks_forecasts():
    periods, data_end = synthetic_periods(), pd.Timestamp("2026-08-26", tz="UTC")
    first_week_start, second_week_start = PERIOD[0], PERIOD[0] + pd.Timedelta(days=7)
    original = build_grid(periods, START, PERIOD[1], data_end, UNITS)
    changed = build_grid(periods, START, PERIOD[1], data_end, UNITS, label_periods=labels_cut_at(periods, first_week_start))
    models = ["historical_rate", "logistic"]
    a = run(original, PERIOD, models, horizons=(1, 6), train_start=START)
    b = run(changed, PERIOD, models, horizons=(1, 6), train_start=START)
    first = lambda p: p[p.week == first_week_start].p.to_numpy()  # noqa: E731
    second = lambda p: p[p.week == second_week_start].p.to_numpy()  # noqa: E731
    np.testing.assert_array_equal(first(a), first(b))  # trained before the changed outcomes
    assert not np.array_equal(second(a), second(b))  # the second week's training does see them


def test_weeks_keep_a_final_partial_week():
    start, end = pd.Timestamp("2026-08-03", tz="UTC"), pd.Timestamp("2026-08-13", tz="UTC")
    assert weeks(start, end) == [(start, start + pd.Timedelta(days=7)), (start + pd.Timedelta(days=7), end)]
