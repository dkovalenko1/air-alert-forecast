"""The single test run (DESIGN.md section 3): the configuration chosen on validation, run once on the test period.

The configuration is read from the experiments' decisions (reports/validation/experiments/decisions.json): training
start (E1), shared or per unit (E2), LightGBM settings (E3), Telegram or not (E4, E5). Each model also runs with the
targeted Telegram set, the ablation fixed in E5's protocol, and the baselines run alongside. The run refuses to
start if its report folder already has results: the test period is used once. To reproduce it, write to another
folder (`test-run --out reports/test-rerun`). Each run also lists its input files with their hashes
(data_manifest.csv). `report` adds day-resampled intervals for the differences between models, a per-week table and
per-raion metrics, from the saved predictions (no model is run again).
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airalert import backtest
from airalert.experiments import OUT, chosen_factory
from airalert.features import KYIV
from airalert.models import HistoricalRate, Persistence, RecentRate, RecentRateByState
from airalert.settings import RAW_DIR, ROOT

REPORT = ROOT / "reports" / "test"
SUFFIX = {None: "", "targeted": " + Telegram (targeted)", "all": " + Telegram (all)"}


def telegram_choice(decisions: dict, model: str) -> str | None:
    """The Telegram set chosen for `model` on validation: E5's targeted set, else E4's full set, else none."""
    if decisions["e5_telegram_targeted"][model] != "without":
        return "targeted"
    if decisions["e4_telegram"][model] != "without":
        return "all"
    return None


def configuration(decisions: dict) -> tuple[pd.Timestamp, dict, dict]:
    """Training start, {label: factory} for every model in the run, and {model: label of its chosen variant}."""
    starts = set(decisions["e1_training_start"].values())
    if len(starts) != 1:
        raise SystemExit(f"E1 chose different training starts per model ({sorted(starts)}); the test run uses one.")
    factories = {"persistence": Persistence, "historical_rate": HistoricalRate, "recent_rate": RecentRate,
                 "recent_rate_by_state": RecentRateByState}
    chosen = {}
    for model in ("logistic", "lightgbm"):
        choice = telegram_choice(decisions, model)
        for telegram in dict.fromkeys([choice, None, "targeted"]):  # the chosen variant first, no duplicates
            factories[model + SUFFIX[telegram]] = chosen_factory(model, decisions, telegram)
        chosen[model] = model + SUFFIX[choice]
    return pd.Timestamp(starts.pop(), tz=KYIV), factories, chosen


def summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean over the six horizons, per model: Brier for all rows and for units calm at t, log loss, ROC-AUC."""
    main = metrics[(metrics.subset == "all") & (metrics.units == "all")].groupby("model")
    calm = metrics[(metrics.subset == "calm at t") & (metrics.units == "all")].groupby("model")
    return pd.DataFrame({"mean_brier": main.brier.mean(), "mean_brier_calm": calm.brier.mean(),
                         "mean_log_loss": main.log_loss.mean(), "mean_roc_auc": main.roc_auc.mean()}).sort_values("mean_brier")


def data_manifest(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Every downloaded Telegram file with its message count, date range and SHA-256. Telegram returns each message's
    latest text, so a later download of the same months can differ (edits, deleted messages); compare hashes."""
    rows = []
    for path in sorted((raw_dir / "telegram").glob("*/*.jsonl")):
        data = path.read_bytes()
        dates = [json.loads(line)["date"] for line in data.decode("utf-8").split("\n") if line]
        rows.append({"file": path.relative_to(raw_dir).as_posix(), "messages": len(dates),
                     "first": min(dates, default=None), "last": max(dates, default=None),
                     "sha256": hashlib.sha256(data).hexdigest()})
    return pd.DataFrame(rows)


def main(out=REPORT) -> None:
    if (out / "metrics.csv").exists():
        raise SystemExit(f"{out / 'metrics.csv'} exists: the test period is used once. To reproduce it, "
                         "write to another folder: test-run --out reports/test-rerun")
    decisions = json.loads((OUT / "decisions.json").read_text())
    train_start, factories, chosen = configuration(decisions)
    out.mkdir(parents=True, exist_ok=True)
    (out / "configuration.json").write_text(json.dumps({
        "period": [f"{d:%Y-%m-%d}" for d in backtest.PERIODS["test"]], "train_start": f"{train_start:%Y-%m-%d}",
        "models": list(factories), "chosen": chosen, "decisions": decisions,
    }, indent=2, ensure_ascii=False) + "\n")
    data_manifest().to_csv(out / "data_manifest.csv", index=False)
    metrics = backtest.main("test", factories, train_start=train_start, telegram=True, report_dir=out)
    table = summary(metrics)
    table.to_csv(out / "summary.csv")
    print(f"\nChosen on validation: {', '.join(chosen.values())}\nMean over the six horizons:\n{table.round(5).to_string()}")


def comparisons(models: list[str], chosen: dict) -> list[tuple[str, str]]:
    """Each model against the chosen LightGBM, and each other Telegram variant against its model without Telegram."""
    main = chosen["lightgbm"]
    pairs = [(m, main) for m in models if m != main]
    return pairs + [(m, m.split(" + ")[0]) for m in models if " + Telegram" in m and m.split(" + ")[0] != main]


def differences(predictions: pd.DataFrame, pairs: list[tuple[str, str]], n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Mean Brier over the six horizons of model a minus model b (> 0: b is better), with a 95% interval from
    resampling whole days (Kyiv time). Neighbouring days are not independent, so the interval is a rough guide."""
    day = predictions.t.dt.tz_convert(KYIV).dt.date
    daily = (predictions.assign(day=day, se=(predictions.p - predictions.y) ** 2)
             .groupby(["model", "horizon", "day"]).se.agg(["sum", "count"]))
    days = sorted(day.unique())
    samples = np.random.default_rng(seed).integers(len(days), size=(n, len(days)))
    everything = np.arange(len(days))[None]

    def score(model, sample):
        wide = daily.loc[model].unstack("horizon").reindex(days)
        totals, counts = wide["sum"].to_numpy()[sample].sum(axis=1), wide["count"].to_numpy()[sample].sum(axis=1)
        return (totals / counts).mean(axis=1)

    rows = []
    for a, b in pairs:
        resampled = score(a, samples) - score(b, samples)
        rows.append({"a": a, "b": b, "difference": (score(a, everything) - score(b, everything))[0],
                     "low": np.percentile(resampled, 2.5), "high": np.percentile(resampled, 97.5),
                     "share_b_better": (resampled > 0).mean()})
    return pd.DataFrame(rows)


def differences_by_horizon(predictions: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """The same comparison for each horizon on its own, for all rows and for units calm at t."""
    rows = []
    for subset, part in (("all", predictions), ("calm at t", predictions[predictions.on_alert == 0])):
        for h, at_h in part.groupby("horizon"):
            rows.append(differences(at_h, pairs).assign(subset=subset, horizon=h))
    return pd.concat(rows, ignore_index=True)[["a", "b", "subset", "horizon", "difference", "low", "high", "share_b_better"]]


def within_units(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per model, horizon and subset (all rows, units calm at t): ROC-AUC pooled over all units and averaged within
    each unit, and how far each unit's mean forecast is from its alert share. Pooled ROC-AUC also rewards telling
    risky raions from safe ones; the within-unit one asks only whether the forecast picks the right hours."""
    rows = []
    for subset, part in (("all", predictions), ("calm at t", predictions[predictions.on_alert == 0])):
        for (model, h), group in part.groupby(["model", "horizon"]):
            units = group.groupby("unit_id")
            auc = units.apply(lambda u: roc_auc_score(u.y, u.p) if u.y.nunique() == 2 else np.nan, include_groups=False)
            gap = units.p.mean() - units.y.mean()
            rows.append({"model": model, "horizon": h, "subset": subset, "pooled_roc_auc": roc_auc_score(group.y, group.p),
                         "within_unit_roc_auc": auc.mean(), "units_with_both_outcomes": int(auc.notna().sum()),
                         "mean_abs_unit_gap": gap.abs().mean(), "largest_unit_gap": gap[gap.abs().idxmax()],
                         "largest_gap_unit": gap.abs().idxmax()})
    return pd.DataFrame(rows)


def report(out=REPORT) -> None:
    predictions = pd.read_parquet(backtest.predictions_path(out))
    config = json.loads((out / "configuration.json").read_text())
    table = differences(predictions, comparisons(config["models"], config["chosen"]))
    table.to_csv(out / "differences.csv", index=False)
    by_horizon = differences_by_horizon(predictions, [(a, b) for a, b in comparisons(config["models"], config["chosen"])
                                                      if " + Telegram" not in a])
    by_horizon.to_csv(out / "differences_by_horizon.csv", index=False)
    weekly = (predictions.assign(se=(predictions.p - predictions.y) ** 2).groupby(["week", "model"]).se.mean()
              .unstack("model")[config["models"]])
    weekly.to_csv(out / "weekly.csv")
    units = within_units(predictions[~predictions.model.str.contains(" + Telegram", regex=False)])
    units.to_csv(out / "within_units.csv", index=False)
    print(f"Mean Brier differences, a − b (> 0: b is better), 95% day-resampled interval:\n{table.round(5).to_string()}")
    print(f"\nBrier by week (all horizons pooled):\n{weekly.round(4).to_string()}")
    recent = by_horizon[by_horizon.a == "recent_rate"]
    print(f"\nRecent rate − LightGBM by horizon (> 0: LightGBM is better):\n{recent.round(4).to_string(index=False)}")
    lightgbm = units[(units.model == config["chosen"]["lightgbm"])]
    print(f"\nLightGBM, ROC-AUC pooled and within units:\n{lightgbm.round(3).to_string(index=False)}")
