"""The single test run (DESIGN.md section 3): the configuration chosen on validation, run once on the test period.

The configuration is read from the experiments' decisions (reports/validation/experiments/decisions.json): training
start (E1), shared or per unit (E2), LightGBM settings (E3), Telegram or not (E4, E5). Each model also runs with the
targeted Telegram set, the ablation fixed in E5's protocol, and both baselines run alongside. The run refuses to
start if the test report already exists: the test period is used once.
"""

import json

import pandas as pd

from airalert import backtest
from airalert.experiments import OUT, chosen_factory
from airalert.features import KYIV
from airalert.models import HistoricalRate, Persistence
from airalert.settings import ROOT

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
    factories, chosen = {"persistence": Persistence, "historical_rate": HistoricalRate}, {}
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


def main() -> None:
    if (REPORT / "metrics.csv").exists():
        raise SystemExit(f"{REPORT / 'metrics.csv'} exists: the test period is used once. Delete it only to rerun on purpose.")
    decisions = json.loads((OUT / "decisions.json").read_text())
    train_start, factories, chosen = configuration(decisions)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "configuration.json").write_text(json.dumps({
        "period": [f"{d:%Y-%m-%d}" for d in backtest.PERIODS["test"]], "train_start": f"{train_start:%Y-%m-%d}",
        "models": list(factories), "chosen": chosen, "decisions": decisions,
    }, indent=2, ensure_ascii=False) + "\n")
    metrics = backtest.main("test", factories, train_start=train_start, telegram=True)
    table = summary(metrics)
    table.to_csv(REPORT / "summary.csv")
    print(f"\nChosen on validation: {', '.join(chosen.values())}\nMean over the six horizons:\n{table.round(5).to_string()}")
