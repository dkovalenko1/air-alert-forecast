"""Calibrate the missing all-clear rule (Q17, Q20) from the official channel: writes reference/max_silence_hours.csv.

An alert in the label channel whose place gets no message for longer than its oblast's limit probably lost its
all-clear. The limit is twice the 99th percentile of official alert lengths in that oblast, at least 12 hours:
- the official channel (@air_alert_ua, rebuilt by `build-labels air_alert_ua`) is the independent reference;
- only alerts before the validation period are used, so validation and test stay unseen;
- the 99th, not the 99.9th, percentile: the official record has a few broken alerts of its own (100-135 h in
  western oblasts), which would inflate the highest percentiles;
- below 12 hours our alerts outside the frontline agree with the official ones 97-99% of the time, so shorter
  limits would throw away mostly real alerts.

Usage: uv run python scripts/calibrate_missing_all_clear.py
"""

import pandas as pd

from airalert.backtest import PERIODS
from airalert.labels import INTERIM_DIR
from airalert.units import REGISTRY

OUT = REGISTRY.parent / "max_silence_hours.csv"
START = pd.Timestamp("2023-01-01", tz="Europe/Kyiv")
FACTOR, FLOOR_HOURS = 2.0, 12.0


def main() -> None:
    official = pd.read_parquet(INTERIM_DIR / "air_alert_ua_label_periods.parquet")
    end = PERIODS["validation"][0]
    official = official[(official.started_at >= START) & (official.finished_at < end)]
    hours = (official.finished_at - official.started_at).dt.total_seconds() / 3600
    p99 = hours.groupby(official.unit_id.str.split("/").str[0]).quantile(0.99)
    table = pd.DataFrame({"oblast": p99.index, "official_p99_hours": p99.round(2).to_numpy(),
                          "max_silence_hours": (p99 * FACTOR).clip(lower=FLOOR_HOURS).round(1).to_numpy()})
    table.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name} from {len(official):,} official alerts, {START:%Y-%m-%d} to {end:%Y-%m-%d}:")
    print(table.sort_values("max_silence_hours", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
