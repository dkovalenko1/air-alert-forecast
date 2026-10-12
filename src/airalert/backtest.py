"""Walk-forward backtest (DESIGN.md sections 3 and 6, Q9).

For every week of the evaluation period, each model is trained on rows known before the week starts and then
forecasts every hour of that week. A training row for time t and horizon h is only used if its label hour
ended by the cutoff (t + h hours <= cutoff), so no outcome after the cutoff reaches training.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from airalert.features import HORIZONS, HOUR, KYIV
from airalert.models import MODELS, feature_columns

# Pinned dates (Q4, Q9), Kyiv time.
TRAIN_START = pd.Timestamp("2025-08-01", tz=KYIV)
PERIODS = {
    "validation": (pd.Timestamp("2026-08-17", tz=KYIV), pd.Timestamp("2026-09-14", tz=KYIV)),
    "test": (pd.Timestamp("2026-09-14", tz=KYIV), pd.Timestamp("2026-10-12", tz=KYIV)),
}


def horizon_frame(grid: pd.DataFrame, h: int) -> pd.DataFrame:
    """The grid with horizon-h columns renamed to the generic names the models use."""
    renamed = grid.rename(columns={
        f"target_hour_h{h}": "target_hour", f"target_weekday_h{h}": "target_weekday",
        f"target_active_day_ago_h{h}": "target_active_day_ago", f"target_active_week_ago_h{h}": "target_active_week_ago",
        f"label_h{h}": "label",
    })
    return renamed[["t", "label", *feature_columns()]]


def weeks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    edges = pd.date_range(start, end, freq="7D")
    return list(zip(edges[:-1], edges[1:]))


def run(grid: pd.DataFrame, period: str, model_names, horizons=HORIZONS, train_start=TRAIN_START) -> pd.DataFrame:
    """Predictions for every (model, horizon, unit, hour) of the period."""
    start, end = PERIODS[period] if isinstance(period, str) else period
    week_list = weeks(start, end)
    total_fits = len(week_list) * len(horizons) * len(model_names)
    results, done, started = [], 0, time.monotonic()
    for week_start, week_end in week_list:
        for h in horizons:
            frame = horizon_frame(grid, h)
            train = frame[(frame.t >= train_start) & (frame.t + h * HOUR <= week_start) & frame.label.notna()]
            test = frame[(frame.t >= week_start) & (frame.t < week_end) & frame.label.notna()]
            assert (train.t + h * HOUR <= week_start).all()  # the label cutoff
            for name in model_names:
                model = MODELS[name]().fit(train, train.label.to_numpy(dtype=int))
                results.append(pd.DataFrame({
                    "model": name, "horizon": h, "week": week_start, "unit_id": test.unit_id.astype(str).to_numpy(),
                    "t": test.t.to_numpy(), "on_alert": test.on_alert.to_numpy(), "y": test.label.to_numpy(dtype=int),
                    "p": np.clip(model.predict(test), 0, 1),
                }))
                done += 1
                if done == 1 or done % 10 == 0:
                    per_fit = (time.monotonic() - started) / done
                    print(f"  {done}/{total_fits} fits, about {per_fit * (total_fits - done) / 60:.0f} min left", flush=True)
    return pd.concat(results, ignore_index=True)


def near_permanent_units(grid: pd.DataFrame, start, end, threshold=0.9) -> set[str]:
    """Units on alert in at least 90% of the period's hours (Q8). Decided on the validation period only."""
    period = grid[(grid.t >= start) & (grid.t < end)]
    share = period.groupby("unit_id", observed=True)["on_alert"].mean()
    return set(share[share >= threshold].index.astype(str))


def _metrics(y, p) -> dict:
    p_clip = np.clip(p, 1e-4, 1 - 1e-4)
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "base_rate": float(np.mean(y)), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p_clip, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p) if both else np.nan, "pr_auc": average_precision_score(y, p) if both else np.nan,
    }


def score(predictions: pd.DataFrame, exclude_units=frozenset()) -> pd.DataFrame:
    """Metrics per model and horizon, for all rows and separately for units calm / on alert at t."""
    rows = []
    kept = predictions[~predictions.unit_id.isin(exclude_units)]
    for (model, h), group in kept.groupby(["model", "horizon"]):
        for subset, part in (("all", group), ("calm at t", group[group.on_alert == 0]), ("on alert at t", group[group.on_alert == 1])):
            if len(part):
                rows.append({"model": model, "horizon": h, "subset": subset, **_metrics(part.y.to_numpy(), part.p.to_numpy())})
    return pd.DataFrame(rows)


def main(period: str, model_names, channel: str = "UkraineAlarmSignal", horizons=HORIZONS, train_start=TRAIN_START) -> None:
    """Build the grid from the label periods, run the walk-forward backtest and write predictions and metrics."""
    from airalert.features import build_grid
    from airalert.labels import INTERIM_DIR
    from airalert.settings import ROOT

    merged = pd.read_parquet(INTERIM_DIR / f"{channel}_merged.parquet")
    data_end = merged.data_end.iloc[0]
    start, end = PERIODS[period]
    if data_end < end + 24 * HOUR:
        print(f"Warning: data ends {data_end:%Y-%m-%d %H:%M} UTC, before the {period} period's last labels.")
    grid = build_grid(merged, train_start - 168 * HOUR, end, data_end)
    print(f"{period}: {start:%Y-%m-%d} → {end:%Y-%m-%d}, {len(grid):,} grid rows, models: {', '.join(model_names)}", flush=True)
    predictions = run(grid, period, model_names, horizons, train_start)

    results_dir, report_dir = ROOT / "data" / "results", ROOT / "reports" / period
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(results_dir / f"{period}_predictions.parquet", index=False)
    permanent = near_permanent_units(grid, *PERIODS["validation"])  # chosen on validation only (Q8)
    metrics = pd.concat([score(predictions).assign(units="all"),
                         score(predictions, exclude_units=permanent).assign(units="without near-permanent")])
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    summary = metrics[(metrics.subset == "all") & (metrics.units == "all")].pivot(index="horizon", columns="model", values="brier")
    print(f"Near-permanent units (validation): {len(permanent)}\nBrier score by horizon (lower is better):\n{summary.round(4)}")
