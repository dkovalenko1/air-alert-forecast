"""How does the model do on the hours our labels mark unknown? (README "Limitations")

Hours after a missing all-clear, or before an all-clear for a never-announced alert, have an unknown label. They are
neither trained on nor scored, so the reported scores say nothing about them. Where the official channel
(`@air_alert_ua`, independent of our label channel) gives an outcome for such an hour, this script scores the chosen
model on it and compares with the normally scored rows. The official record is left out during its own outages and
ends 2026-09-07, so this covers part of the validation period and only some of the unknown hours.

Same weekly walk-forward and model as the backtest, validation period only. Writes
reports/validation/unknown_hours_check.csv.

Usage: uv run python scripts/unknown_hours_check.py
"""

import json

import numpy as np
import pandas as pd

from airalert import backtest
from airalert.experiments import OUT, chosen_factory
from airalert.features import HORIZONS, HOUR, KYIV, build_grid
from airalert.labels import INTERIM_DIR, read_channel
from airalert.settings import ROOT
from airalert.units import forecast_unit_ids
from airalert.validation import outages


def official_outcomes(start, end) -> pd.DataFrame:
    """(unit_id, t, horizon) → the official record's label, for hours it covers outside its outages."""
    periods = pd.read_parquet(INTERIM_DIR / "air_alert_ua_label_periods.parquet")
    data_end = periods.data_end.iloc[0]
    grid = build_grid(periods[["unit_id", "started_at", "finished_at"]], start, end, data_end, forecast_unit_ids())
    gaps = outages(m["date"] for m in read_channel("air_alert_ua"))
    parts = []
    for h in HORIZONS:
        part = grid[["unit_id", "t", f"label_h{h}"]].rename(columns={f"label_h{h}": "outcome"}).assign(horizon=h)
        target = part.t + (h - 1) * HOUR  # start of the forecast hour; leave out hours touching an outage
        in_outage = np.zeros(len(part), dtype=bool)
        for a, b in gaps:
            in_outage |= ((target + HOUR > a) & (target < b)).to_numpy()
        parts.append(part[~in_outage].dropna(subset=["outcome"]))
    out = pd.concat(parts, ignore_index=True)
    return out.assign(unit_id=out.unit_id.astype(str))


def main() -> None:
    decisions = json.loads((OUT / "decisions.json").read_text())
    train_start = pd.Timestamp(decisions["e1_training_start"]["lightgbm"], tz=KYIV)
    factory = chosen_factory("lightgbm", decisions)
    grid = backtest.load_grid("validation", train_start)
    start, end = backtest.PERIODS["validation"]
    reference = official_outcomes(start, end).set_index(["unit_id", "t", "horizon"]).outcome
    rows = []
    for week_start, week_end in backtest.weeks(start, end):
        for h in HORIZONS:
            frame = backtest.horizon_frame(grid, h)
            train, y = backtest.training_rows(frame, h, train_start, week_start)
            model = factory().fit(train, y)
            window = frame[(frame.t >= week_start) & (frame.t < week_end) & (frame.t + h * HOUR <= end)]
            window = window.assign(p=np.clip(model.predict(window), 0, 1), horizon=h, unit_id=window.unit_id.astype(str))
            key = pd.MultiIndex.from_frame(window[["unit_id", "t", "horizon"]])
            window["official"] = reference.reindex(key).to_numpy()
            rows.append(window[["horizon", "label", "official", "p"]])
        print(f"week of {week_start:%Y-%m-%d} done", flush=True)
    data = pd.concat(rows, ignore_index=True)
    known = data[data.label.notna()]
    unknown = data[data.label.isna() & data.official.notna()]
    out = []
    for h in HORIZONS:
        a, b = known[known.horizon == h], unknown[unknown.horizon == h]
        brier = lambda d, col: float(np.mean((d.p - d[col]) ** 2)) if len(d) else np.nan  # noqa: E731
        out.append({"horizon": h, "scored_rows": len(a), "scored_brier": brier(a, "label"), "unknown_rows_with_official_outcome": len(b),
                    "unknown_brier": brier(b, "official"), "unknown_alert_share": float(b.official.mean()) if len(b) else np.nan,
                    "mean_forecast_on_unknown": float(b.p.mean()) if len(b) else np.nan})
    result = pd.DataFrame(out)
    result.to_csv(ROOT / "reports" / "validation" / "unknown_hours_check.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
