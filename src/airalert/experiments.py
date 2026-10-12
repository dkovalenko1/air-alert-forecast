"""Validation experiments E1–E3 (DESIGN.md section 3, protocol fixed before they ran).

Each experiment runs the weekly walk-forward backtest on the validation period for a set of variants, writes every
variant's metrics to reports/validation/experiments/<experiment>.csv, and applies the selection rule: the lowest
Brier score averaged over the six horizons (all rows, all units) wins, unless it beats the default by less than
MIN_GAIN, in which case the default is kept.
"""

import functools
import json

import pandas as pd

from airalert.backtest import PERIODS, load_grid, near_permanent_units, run, score
from airalert.features import KYIV
from airalert.models import LightGBM, Logistic, PerUnit
from airalert.settings import ROOT

MIN_GAIN = 0.0005
OUT = ROOT / "reports" / "validation" / "experiments"
STARTS = {"2023-01-01": "2023-01-01", "2025-08-01": "2025-08-01", "2025-11-01": "2025-11-01"}
LIGHTGBM_SETTINGS = {
    "default": {},
    "small trees": {"num_leaves": 15},
    "large trees": {"num_leaves": 255, "min_child_samples": 200},
    "early stopping": {"early_stopping_days": 14, "n_estimators": 3000},
    "slow early stopping": {"early_stopping_days": 14, "n_estimators": 5000, "learning_rate": 0.02},
    "no unit identity": {"use_unit": False},
}


def _evaluate(grid, factories, train_start) -> pd.DataFrame:
    predictions = run(grid, "validation", factories, train_start=train_start)
    permanent = near_permanent_units(grid, *PERIODS["validation"])
    return pd.concat([score(predictions).assign(units="all"),
                      score(predictions, exclude_units=permanent).assign(units="without near-permanent")])


def choose(metrics: pd.DataFrame, default: str, variant_column: str = "variant") -> tuple[str, pd.DataFrame]:
    """Apply the selection rule. Returns the winner and the summary table it was chosen from."""
    main = metrics[(metrics.subset == "all") & (metrics.units == "all")]
    calm = metrics[(metrics.subset == "calm at t") & (metrics.units == "all")]
    summary = pd.DataFrame({
        "mean_brier": main.groupby(variant_column).brier.mean(),
        "mean_brier_calm": calm.groupby(variant_column).brier.mean(),
        "mean_log_loss": main.groupby(variant_column).log_loss.mean(),
        "mean_roc_auc": main.groupby(variant_column).roc_auc.mean(),
    }).sort_values("mean_brier")
    best = summary.index[0]
    winner = best if summary.loc[default, "mean_brier"] - summary.loc[best, "mean_brier"] >= MIN_GAIN else default
    return winner, summary


def _save(name: str, metrics: pd.DataFrame, results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / f"{name}.csv", index=False)
    decisions = OUT / "decisions.json"
    known = json.loads(decisions.read_text()) if decisions.exists() else {}
    known[name] = results
    decisions.write_text(json.dumps(known, indent=2, ensure_ascii=False) + "\n")


def _report(title: str, per_model: dict) -> None:
    print(f"\n== {title}")
    for model, (winner, summary) in per_model.items():
        print(f"-- {model}: winner = {winner}\n{summary.round(5).to_string()}")


def training_start() -> dict:
    """E1: which training start gives the best validation scores (Q9)?"""
    parts = []
    for label, start in STARTS.items():
        train_start = pd.Timestamp(start, tz=KYIV)
        grid = load_grid("validation", train_start)
        print(f"E1 {label}: {len(grid):,} grid rows", flush=True)
        metrics = _evaluate(grid, {"logistic": Logistic, "lightgbm": LightGBM}, train_start)
        parts.append(metrics.assign(variant=label))
        del grid
    metrics = pd.concat(parts, ignore_index=True)
    per_model = {m: choose(metrics[metrics.model == m], "2025-08-01") for m in ("logistic", "lightgbm")}
    _report("E1 training start", per_model)
    _save("e1_training_start", metrics, {m: w for m, (w, _) in per_model.items()})
    return {m: w for m, (w, _) in per_model.items()}


def _start(model: str) -> pd.Timestamp:
    decisions = json.loads((OUT / "decisions.json").read_text())
    return pd.Timestamp(decisions["e1_training_start"][model], tz=KYIV)


def shared_or_per_unit() -> dict:
    """E2: one model for all units, or one per unit (Q13)? Each model uses its E1 training start."""
    parts, chosen = [], {}
    for model, base in (("logistic", Logistic), ("lightgbm", LightGBM)):
        train_start = _start(model)
        grid = load_grid("validation", train_start)
        factories = {"shared": base, "per unit": functools.partial(PerUnit, base)}
        metrics = pd.concat([_evaluate(grid, {k: v}, train_start).assign(variant=k, model=model) for k, v in factories.items()])
        parts.append(metrics)
        chosen[model] = choose(metrics, "shared")
        del grid
    _report("E2 shared or per unit", chosen)
    _save("e2_shared_or_per_unit", pd.concat(parts, ignore_index=True), {m: w for m, (w, _) in chosen.items()})
    return {m: w for m, (w, _) in chosen.items()}


def lightgbm_settings() -> dict:
    """E3: which LightGBM settings? Uses the E1 training start and the E2 choice for LightGBM."""
    decisions = json.loads((OUT / "decisions.json").read_text())
    train_start = _start("lightgbm")
    per_unit = decisions["e2_shared_or_per_unit"]["lightgbm"] == "per unit"
    grid = load_grid("validation", train_start)
    parts = []
    for label, params in LIGHTGBM_SETTINGS.items():
        factory = functools.partial(LightGBM, **params)
        if per_unit:
            factory = functools.partial(PerUnit, factory)
        parts.append(_evaluate(grid, {"lightgbm": factory}, train_start).assign(variant=label))
        print(f"E3 {label}: done", flush=True)
    metrics = pd.concat(parts, ignore_index=True)
    winner, summary = choose(metrics, "default")
    _report("E3 LightGBM settings", {"lightgbm": (winner, summary)})
    _save("e3_lightgbm_settings", metrics, {"lightgbm": winner})
    return {"lightgbm": winner}


EXPERIMENTS = {"training-start": training_start, "per-unit": shared_or_per_unit, "lightgbm": lightgbm_settings}
