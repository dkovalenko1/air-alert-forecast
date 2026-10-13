"""Validation experiments E1–E5 (DESIGN.md section 3, each protocol fixed before it ran).

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
        # per unit: 119 small models trained side by side, one thread each (same models, only faster)
        one_thread = functools.partial(LightGBM, n_jobs=1) if base is LightGBM else base
        factories = {"shared": base, "per unit": functools.partial(PerUnit, one_thread)}
        metrics = pd.concat([_evaluate(grid, {k: v}, train_start).assign(variant=k, model=model) for k, v in factories.items()])
        parts.append(metrics)
        chosen[model] = choose(metrics, "shared")
        del grid
    _report("E2 shared or per unit", chosen)
    _save("e2_shared_or_per_unit", pd.concat(parts, ignore_index=True), {m: w for m, (w, _) in chosen.items()})
    return {m: w for m, (w, _) in chosen.items()}


def lightgbm_mode(decisions: dict, assume: str | None) -> str:
    """'shared' or 'per unit' for E3: E2's decision, or an assumption made while E2 is still running.

    An assumption that contradicts E2's decision is refused: E3 would then have tested the wrong setup.
    """
    decided = decisions.get("e2_shared_or_per_unit", {}).get("lightgbm")
    if assume and decided and assume != decided:
        raise SystemExit(f"E2 chose '{decided}' for LightGBM; E3 cannot assume '{assume}'.")
    if not (assume or decided):
        raise SystemExit("E2 has not decided yet: run it first, or start E3 early with --assume-e2 shared|'per unit'.")
    return decided or assume


def lightgbm_settings(assume_e2: str | None = None) -> dict:
    """E3: which LightGBM settings? Uses the E1 training start and the E2 choice for LightGBM.

    `assume_e2` lets E3 start while E2 is still running; the assumption is saved with the result and checked
    against E2's decision (`check_e3_assumption`) before E3's winner is used.
    """
    decisions = json.loads((OUT / "decisions.json").read_text())
    mode = lightgbm_mode(decisions, assume_e2)
    train_start = _start("lightgbm")
    grid = load_grid("validation", train_start)
    parts = []
    for label, params in LIGHTGBM_SETTINGS.items():
        factory = functools.partial(LightGBM, **params)
        if mode == "per unit":
            factory = functools.partial(PerUnit, functools.partial(LightGBM, n_jobs=1, **params))
        parts.append(_evaluate(grid, {"lightgbm": factory}, train_start).assign(variant=label, e2_mode=mode))
        print(f"E3 {label}: done", flush=True)
    metrics = pd.concat(parts, ignore_index=True)
    winner, summary = choose(metrics, "default")
    _report(f"E3 LightGBM settings ({mode})", {"lightgbm": (winner, summary)})
    result = {"lightgbm": winner, "e2_mode": mode, "e2_assumed": assume_e2 is not None and "e2_shared_or_per_unit" not in decisions}
    _save("e3_lightgbm_settings", metrics, result)
    return result


def check_e3_assumption() -> None:
    """After E2 has decided: does the E3 result rest on the right setup?"""
    decisions = json.loads((OUT / "decisions.json").read_text())
    e2, e3 = decisions["e2_shared_or_per_unit"]["lightgbm"], decisions["e3_lightgbm_settings"]
    status = "valid" if e3["e2_mode"] == e2 else "INVALID: rerun E3"
    print(f"E2 chose '{e2}' for LightGBM; E3 was run as '{e3['e2_mode']}': {status}")


def require_valid_e3(decisions: dict) -> dict:
    """E3's result, refused if E3 ran on a different shared/per-unit setup than E2 chose."""
    e2, e3 = decisions["e2_shared_or_per_unit"]["lightgbm"], decisions["e3_lightgbm_settings"]
    if e3["e2_mode"] != e2:
        raise SystemExit(f"E3 ran as '{e3['e2_mode']}' but E2 chose '{e2}' for LightGBM: rerun E3 first.")
    return e3


def chosen_factory(model: str, decisions: dict, telegram: str | None = None):
    """'logistic' or 'lightgbm' as chosen by E2 and E3, optionally with a set of Telegram features."""
    per_unit = decisions["e2_shared_or_per_unit"][model] == "per unit"
    if model == "logistic":
        base = functools.partial(Logistic, telegram=telegram)
    else:
        params = {**LIGHTGBM_SETTINGS[require_valid_e3(decisions)["lightgbm"]], **({"n_jobs": 1} if per_unit else {})}
        base = functools.partial(LightGBM, telegram=telegram, **params)
    return functools.partial(PerUnit, base) if per_unit else base


def _without_or_with(name: str, title: str, feature_set: str, label: str) -> dict:
    """Each model as chosen by E1-E3, without (default) and with one set of Telegram features (E4, E5)."""
    decisions = json.loads((OUT / "decisions.json").read_text())
    parts, chosen = [], {}
    for model in ("logistic", "lightgbm"):
        train_start = _start(model)
        grid = load_grid("validation", train_start, telegram=True)
        variants = {"without": None, label: feature_set}
        metrics = pd.concat([_evaluate(grid, {model: chosen_factory(model, decisions, telegram)}, train_start)
                             .assign(variant=variant) for variant, telegram in variants.items()])
        parts.append(metrics)
        chosen[model] = choose(metrics, "without")
        del grid
    _report(title, chosen)
    metrics = pd.concat(parts, ignore_index=True)
    _save(name, metrics, {m: w for m, (w, _) in chosen.items()})
    main = metrics[(metrics.units == "all") & (metrics.subset.isin(["all", "calm at t"]))]
    print(main.pivot_table(index=["model", "subset", "horizon"], columns="variant", values="brier").round(4).to_string())
    return {m: w for m, (w, _) in chosen.items()}


def telegram_comparison() -> dict:
    """E4 (Q12): without (default) and with all the Telegram threat-post features."""
    return _without_or_with("e4_telegram", "E4 with or without Telegram", "all", "with Telegram")


def telegram_targeted() -> dict:
    """E5 (Q12): without (default) and with the small targeted set of Telegram features."""
    return _without_or_with("e5_telegram_targeted", "E5 with or without the targeted Telegram set", "targeted",
                            "targeted Telegram")


EXPERIMENTS = {"training-start": training_start, "per-unit": shared_or_per_unit, "lightgbm": lightgbm_settings,
               "telegram": telegram_comparison, "telegram-targeted": telegram_targeted}
