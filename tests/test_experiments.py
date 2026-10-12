import pandas as pd

from airalert.experiments import MIN_GAIN, choose


def metrics(briers: dict) -> pd.DataFrame:
    rows = [{"variant": v, "horizon": h, "subset": s, "units": "all", "brier": b, "log_loss": b * 3, "roc_auc": 0.9}
            for v, b in briers.items() for h in (1, 24) for s in ("all", "calm at t")]
    return pd.DataFrame(rows)


def test_clear_winner_is_chosen():
    winner, summary = choose(metrics({"default": 0.120, "other": 0.110}), "default")
    assert winner == "other" and summary.index[0] == "other"


def test_default_is_kept_when_the_gain_is_below_the_threshold():
    winner, _ = choose(metrics({"default": 0.1200, "other": 0.1200 - MIN_GAIN / 2}), "default")
    assert winner == "default"


def test_only_all_rows_and_all_units_decide():
    m = metrics({"default": 0.120, "other": 0.125})
    m.loc[m.subset == "calm at t", "brier"] = m.loc[m.subset == "calm at t", "variant"].map({"default": 0.2, "other": 0.01})
    assert choose(m, "default")[0] == "default"
