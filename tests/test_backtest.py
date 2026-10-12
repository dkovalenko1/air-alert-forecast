import numpy as np
import pandas as pd

from airalert.backtest import near_permanent_units, run, score
from airalert.features import build_grid

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
    grid = build_grid(synthetic_periods(), START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"))
    predictions = run(grid, PERIOD, ["persistence", "historical_rate", "logistic", "lightgbm"], horizons=(1, 6), train_start=START)
    assert set(predictions.model) == {"persistence", "historical_rate", "logistic", "lightgbm"}
    assert predictions.p.between(0, 1).all()
    assert predictions.week.nunique() == 2 and predictions.t.min() >= PERIOD[0] and predictions.t.max() < PERIOD[1]
    metrics = score(predictions).set_index(["model", "horizon", "subset"])
    # At 6 h ahead, the daily rhythm makes the historical rate far better than "same as now".
    assert metrics.loc[("historical_rate", 6, "all"), "brier"] < metrics.loc[("persistence", 6, "all"), "brier"]


def test_near_permanent_units_and_excluding_them():
    grid = build_grid(synthetic_periods(), START, PERIOD[1], pd.Timestamp("2026-08-26", tz="UTC"))
    assert near_permanent_units(grid, *PERIOD) == {"B/d"}
    predictions = run(grid, PERIOD, ["persistence"], horizons=(1,), train_start=START)
    assert score(predictions, exclude_units={"B/d"}).n.sum() < score(predictions).n.sum()
