"""How much does the edited-messages assumption matter? (ADR-0007, README "No leakage")

Telegram returns only a message's latest text, so our labels date every message by its original posting time and read
its final wording. A pessimistic alternative: a message counts only from its last edit (the earliest time its final
wording is certain to have existed). This script rebuilds the alert history that way, then trains and scores the
chosen model on the validation period, once with that history and once with ours:

  - "as built": the project's alert history (features and training labels), the chosen LightGBM;
  - "last-edit": every message dated by its last edit, for features and for training labels;
both scored against the same outcomes (the project's labels) on the rows both can score, so the difference is what
the assumption could cost.

Only messages posted since 2025-10-01 are shifted. The channel-wide re-edits of 2024-01-30 and 2025-06-27 changed old
messages in bulk and say nothing about when their text existed; every message since July 2025 has been edited
automatically soon after posting. This is an upper bound on the harm, not an estimate: it also delays every alert
start by the edit delay, although the original posting time is when the alert was announced.

Validation period only; the test period is not loaded. Writes reports/validation/provenance_replay.csv.

Usage: uv run python scripts/replay_edit_times.py
"""

import json

import numpy as np
import pandas as pd

from airalert import backtest
from airalert.experiments import OUT, chosen_factory
from airalert.features import HORIZONS, HOUR, KYIV, build_grid
from airalert.labels import (CLASSIFIERS, corrections, known_pieces, max_silence, merge_per_unit, read_channel,
                             unit_alerts_from_messages)
from airalert.settings import ROOT
from airalert.units import forecast_unit_ids

CHANNEL = "UkraineAlarmSignal"
SHIFT_FROM = "2025-10-01"


def last_edit_messages(messages: list[dict]) -> list[dict]:
    """Every message since SHIFT_FROM dated by its last edit (never earlier than its posting), in time order."""
    shifted = []
    for m in messages:
        edited = m.get("edit_date")
        date = max(m["date"], edited) if edited and m["date"] >= SHIFT_FROM else m["date"]
        shifted.append({**m, "date": date})
    return sorted(shifted, key=lambda m: (pd.Timestamp(m["date"]), m["id"]))


def grid_from(messages: list[dict], train_start: pd.Timestamp) -> pd.DataFrame:
    alerts, _, _ = unit_alerts_from_messages(messages, CLASSIFIERS[CHANNEL], max_silence=max_silence)
    data_end = pd.Timestamp(messages[-1]["date"])
    merged = merge_per_unit(alerts[~alerts.missed_start], data_end)
    label_periods = merge_per_unit(known_pieces(alerts), data_end)
    end = backtest.PERIODS["validation"][1]
    return build_grid(merged, train_start - 168 * HOUR, end, data_end, forecast_unit_ids(), label_periods, corrections(alerts))


def truth(grid: pd.DataFrame) -> pd.DataFrame:
    """(unit_id, t, horizon) → label, from the project's own grid."""
    parts = [grid[["unit_id", "t", f"label_h{h}"]].rename(columns={f"label_h{h}": "label"}).assign(horizon=h) for h in HORIZONS]
    out = pd.concat(parts, ignore_index=True)
    return out.assign(unit_id=out.unit_id.astype(str))


def main() -> None:
    decisions = json.loads((OUT / "decisions.json").read_text())
    train_start = pd.Timestamp(decisions["e1_training_start"]["lightgbm"], tz=KYIV)
    factory = {"lightgbm": chosen_factory("lightgbm", decisions)}
    as_built = backtest.load_grid("validation", train_start)
    messages = read_channel(CHANNEL)
    print(f"{len(messages):,} messages; {sum(m['date'] >= SHIFT_FROM and bool(m.get('edit_date')) for m in messages):,} "
          f"dated by their last edit", flush=True)
    last_edit = grid_from(last_edit_messages(messages), train_start)
    outcomes = truth(as_built).rename(columns={"label": "outcome"})
    scored = {}
    for name, grid in (("as built", as_built), ("last-edit", last_edit)):
        predictions = backtest.run(grid, "validation", factory, train_start=train_start)
        merged = predictions.assign(unit_id=predictions.unit_id.astype(str)).drop(columns="y").merge(
            outcomes, on=["unit_id", "t", "horizon"], how="inner").dropna(subset=["outcome"]).rename(columns={"outcome": "y"})
        scored[name] = merged.assign(y=merged.y.astype(int))
    keys = ["unit_id", "t", "horizon"]  # score both on the rows they share, so the comparison is like for like
    common = scored["as built"][keys].merge(scored["last-edit"][keys], on=keys)
    rows = []
    for name, predictions in scored.items():
        table = backtest.score(predictions.merge(common, on=keys))
        table = table[table.subset.isin(["all", "calm at t"])].assign(history=name)
        rows.append(table[["history", "horizon", "subset", "n", "brier", "roc_auc"]])
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(ROOT / "reports" / "validation" / "provenance_replay.csv", index=False)
    wide = result.pivot_table(index=["subset", "horizon"], columns="history", values="brier")
    wide["cost"] = wide["last-edit"] - wide["as built"]
    print(wide.round(4).to_string())
    mean = result.groupby(["history", "subset"]).brier.mean().unstack("history")
    print("\nMean Brier over the six horizons:\n" + mean.assign(cost=mean["last-edit"] - mean["as built"]).round(5).to_string())
    n = result.groupby("history").n.sum()
    print("\nScored rows:", n.to_dict())


if __name__ == "__main__":
    main()
