"""Forecasts from the latest data (DESIGN.md section 1).

`train` fits the configuration chosen on validation (final.configuration) on everything known at a cutoff, one model
per horizon, and saves it to models/. `predict` loads it and prints, for every unit, the probability of an alert in
the hour starting h - 1 hours after the forecast time, for each horizon h. It trains nothing.

A model trained at cutoff C only forecasts at times t >= C, and the data must reach t: so nothing a forecast uses,
in its inputs or in its model, comes from after t (ADR-0001).
"""

import json
import pickle

import pandas as pd

from airalert.backtest import horizon_frame, load_grid, training_rows
from airalert.experiments import OUT
from airalert.features import HORIZONS, HOUR, KYIV
from airalert.final import configuration
from airalert.labels import INTERIM_DIR
from airalert.settings import ROOT

MODELS_DIR = ROOT / "models"


def data_end(channel: str = "UkraineAlarmSignal") -> pd.Timestamp:
    return pd.read_parquet(INTERIM_DIR / f"{channel}_merged.parquet", columns=["data_end"]).data_end.iloc[0]


def check_forecast_time(at: pd.Timestamp, cutoff: pd.Timestamp, end: pd.Timestamp) -> None:
    """Refuse a forecast whose model was trained after it, or whose inputs the data does not reach yet."""
    if at != at.floor("h"):
        raise SystemExit(f"Forecast times are whole hours; got {at.tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv.")
    if at < cutoff:
        raise SystemExit(f"The saved model was trained at {cutoff.tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv, after the forecast "
                         f"time: it may have learnt from what came later. Train with --cutoff at or before the forecast time.")
    if at > end:
        raise SystemExit(f"The data ends at {end.tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv, before the forecast time: "
                         "download and rebuild the labels first.")


def _kyiv_hour(text: str | None, default: pd.Timestamp) -> pd.Timestamp:
    return default if text is None else pd.Timestamp(text, tz=KYIV).tz_convert("UTC")


def train(cutoff: str | None = None) -> None:
    """Fit the chosen model at `cutoff` (Kyiv time; default: the last whole hour of the data) and save it."""
    train_start, factories, chosen = configuration(json.loads((OUT / "decisions.json").read_text()))
    name = chosen["lightgbm"]
    end = data_end()
    at = _kyiv_hour(cutoff, end.floor("h"))
    check_forecast_time(at, at, end)
    grid = load_grid(at, train_start, telegram="Telegram" in name)
    models = {}
    for h in HORIZONS:
        rows, y = training_rows(horizon_frame(grid, h), h, train_start, at)
        models[h] = factories[name]().fit(rows, y)
        print(f"  h = {h:>2}: trained on {len(rows):,} rows", flush=True)
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name.replace(' ', '_')}_{at.tz_convert(KYIV):%Y%m%d_%H%M}.pkl"
    with path.open("wb") as f:
        pickle.dump({"model": name, "cutoff": at, "train_start": train_start, "models": models}, f)
    print(f"Saved {path.relative_to(ROOT)} (trained at {at.tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv)")


def latest_saved(models_dir=MODELS_DIR) -> dict:
    """The saved model with the latest cutoff."""
    paths = list(models_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit("No saved model: run `train` first.")
    saved = []
    for path in paths:
        with path.open("rb") as f:
            saved.append(pickle.load(f))
    return max(saved, key=lambda s: s["cutoff"])


def forecast(at: pd.Timestamp, saved: dict, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Units × horizons: the probability of an alert, plus whether each unit is on alert at `at`. `grid` (default:
    built from the data on disk) must contain the rows at `at`."""
    if grid is None:
        grid = load_grid(at + HOUR, at, telegram="Telegram" in saved["model"])
    now = grid[grid.t == at]
    table = pd.DataFrame({"unit": now.unit_id.astype(str).to_numpy(), "on alert": now.on_alert.to_numpy().astype(bool)})
    for h, model in saved["models"].items():
        table[f"h={h}"] = model.predict(horizon_frame(now, h))
    return table.sort_values("h=1", ascending=False, ignore_index=True)


def predict(at: str | None = None) -> pd.DataFrame:
    """Print the forecast at `at` (Kyiv time; default: the last whole hour of the data) with the latest saved model."""
    saved, end = latest_saved(), data_end()
    t = _kyiv_hour(at, end.floor("h"))
    check_forecast_time(t, saved["cutoff"], end)
    table = forecast(t, saved)
    print(f"Forecast time: {t.tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv (data up to {end.tz_convert(KYIV):%Y-%m-%d %H:%M}); model: "
          f"{saved['model']}, trained at {saved['cutoff'].tz_convert(KYIV):%Y-%m-%d %H:%M} Kyiv")
    print("Probability of an alert in the hour starting h - 1 hours from now:\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.round(2).to_string(index=False))
    return table
