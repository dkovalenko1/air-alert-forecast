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


def test_training_labels_are_as_known_at_the_cutoff(monkeypatch):
    """An alert starts at 22:10 before the cutoff and is retracted at 00:30, after it. At the cutoff it was still
    believed: the model trained then must see exactly what it would have seen with data up to the cutoff."""
    fits = []

    class Spy(MODELS["historical_rate"]):
        def fit(self, frame, y):
            fits.append((frame.drop(columns=["label", "published_label", "label_first_corrected_at", "label_corrected_at"]),
                         np.asarray(y)))
            return super().fit(frame, y)

    monkeypatch.setitem(MODELS, "spy", Spy)
    cutoff, data_end = PERIOD[0], pd.Timestamp("2026-08-26", tz="UTC")
    mistaken = pd.DataFrame({"unit_id": ["A/a"], "started_at": [cutoff - pd.Timedelta(minutes=110)],
                             "finished_at": [cutoff + pd.Timedelta(minutes=30)]})
    periods = pd.concat([synthetic_periods(), mistaken], ignore_index=True)
    retracted_later = build_grid(periods, START, PERIOD[1], data_end, UNITS, label_periods=synthetic_periods(), corrections=mistaken)
    known_at_cutoff = build_grid(periods, START, PERIOD[1], data_end, UNITS)  # not retracted yet
    for grid in (retracted_later, known_at_cutoff):
        run(grid, (cutoff, cutoff + pd.Timedelta(days=7)), ["spy"], horizons=(1,), train_start=START)
    (frame_a, y_a), (frame_b, y_b) = fits
    pd.testing.assert_frame_equal(frame_a, frame_b)
    np.testing.assert_array_equal(y_a, y_b)
    late_evening = (frame_a.unit_id == "A/a").to_numpy() & frame_a.t.isin([cutoff - 2 * HOUR, cutoff - HOUR]).to_numpy()
    assert (y_a[late_evening] == 1).all()  # believed on alert at the cutoff
    # the scored truth is corrected: that alert never happened
    assert retracted_later.set_index(["unit_id", "t"]).loc[("A/a", cutoff - HOUR), "label_h1"] == 0


def test_validation_scores_never_depend_on_the_test_period():
    data_end = pd.Timestamp("2026-08-26", tz="UTC")
    later = pd.DataFrame({"unit_id": ["A/a"], "started_at": [PERIOD[1] + pd.Timedelta(minutes=65)],
                          "finished_at": [PERIOD[1] + pd.Timedelta(minutes=95)]})
    original = build_grid(synthetic_periods(), START, PERIOD[1], data_end, UNITS)
    changed = build_grid(pd.concat([synthetic_periods(), later]), START, PERIOD[1], data_end, UNITS)
    a = run(original, PERIOD, ["persistence", "historical_rate"], horizons=(1, 24), train_start=START)
    b = run(changed, PERIOD, ["persistence", "historical_rate"], horizons=(1, 24), train_start=START)
    pd.testing.assert_frame_equal(a, b)
    assert (pd.to_datetime(a.t, utc=True) + a.horizon * HOUR <= PERIOD[1]).all()


def test_near_permanent_counts_hours_with_any_alert():
    """On alert from minute 1 to 59 of every hour: calm at every whole hour t, yet every hour has an alert."""
    hours = pd.date_range(START - pd.Timedelta(days=8), PERIOD[1] + pd.Timedelta(days=2), freq="h")
    flicker = pd.DataFrame({"unit_id": "B/c", "started_at": hours + pd.Timedelta(minutes=1), "finished_at": hours + pd.Timedelta(minutes=59)})
    grid = build_grid(flicker, START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"), UNITS)
    assert grid[grid.unit_id == "B/c"].on_alert.max() == 0
    assert near_permanent_units(grid, *PERIOD) == {"B/c"}


def test_lost_all_clear_hours_are_unknown_and_known_only_after_detection(monkeypatch):
    """Published: on alert 2026-08-02 20:00 → 08-04 02:00. The all-clear was lost after 21:00; detected 08-03 09:00."""
    data_end = pd.Timestamp("2026-08-26", tz="UTC")
    published = pd.DataFrame({"unit_id": ["A/b"], "started_at": [pd.Timestamp("2026-08-02 20:00", tz="UTC")],
                              "finished_at": [pd.Timestamp("2026-08-04 02:00", tz="UTC")]})
    known = published.assign(finished_at=pd.Timestamp("2026-08-02 21:00", tz="UTC"))
    unknown = pd.DataFrame({"unit_id": ["A/b"], "started_at": [pd.Timestamp("2026-08-02 21:00", tz="UTC")],
                            "finished_at": [pd.Timestamp("2026-08-04 02:00", tz="UTC")],
                            "corrected_at": [pd.Timestamp("2026-08-03 09:00", tz="UTC")], "unknown": [True]})
    periods = pd.concat([synthetic_periods(), published], ignore_index=True)
    grid = build_grid(periods, START, PERIOD[1], data_end, UNITS, pd.concat([synthetic_periods(), known]), unknown).set_index(["unit_id", "t"])
    assert grid.loc[("A/b", pd.Timestamp("2026-08-02 20:00", tz="UTC")), "label_h1"] == 1  # confirmed alert in that hour
    assert np.isnan(grid.loc[("A/b", pd.Timestamp("2026-08-02 23:00", tz="UTC")), "label_h1"])
    assert grid.loc[("A/b", pd.Timestamp("2026-08-02 23:00", tz="UTC")), "published_label_h1"] == 1

    fits = []

    class Spy(MODELS["historical_rate"]):
        def fit(self, frame, y):
            fits.append(dict(zip(frame.t[frame.unit_id == "A/b"], np.asarray(y)[(frame.unit_id == "A/b").to_numpy()])))
            return super().fit(frame, y)

    monkeypatch.setitem(MODELS, "spy", Spy)
    lost = pd.Timestamp("2026-08-02 23:00", tz="UTC")
    run(grid.reset_index(), (pd.Timestamp("2026-08-03 03:00", tz="UTC"), pd.Timestamp("2026-08-03 04:00", tz="UTC")), ["spy"], horizons=(1,), train_start=START)
    run(grid.reset_index(), (pd.Timestamp("2026-08-03 12:00", tz="UTC"), pd.Timestamp("2026-08-03 13:00", tz="UTC")), ["spy"], horizons=(1,), train_start=START)
    before, after = fits
    assert before[lost] == 1  # at 03:00 the alert was still believed on
    assert lost not in after  # by 12:00 the lost all-clear was known: the hour is not trained on


def test_a_label_with_only_some_corrections_known_is_not_trained_on():
    """Two corrections touch one hour: one known before the cutoff, one after. The label as it stood at the cutoff
    is neither the published nor the final one, so it is left out; with none or all known, those are used."""
    from airalert.backtest import label_known_at

    cutoff = pd.Timestamp("2026-08-10", tz="UTC")
    before, after = cutoff - pd.Timedelta(hours=1), cutoff + pd.Timedelta(hours=1)
    frame = pd.DataFrame({"label": [0.0, 0.0, 0.0, 1.0], "published_label": [1.0, 1.0, 1.0, 1.0],
                          "label_first_corrected_at": [before, after, before, pd.NaT],
                          "label_corrected_at": [before, after, after, pd.NaT]})
    np.testing.assert_array_equal(label_known_at(frame, cutoff), [0.0, 1.0, np.nan, 1.0])
