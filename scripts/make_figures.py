"""Figures for the README (docs/figures/*.png).

Exploratory figures are built from the label periods (`build-labels UkraineAlarmSignal` first). They use only data
before the test period (up to the end of validation, 2026-09-13): they informed modelling choices, so the test weeks
stayed unseen. Result figures read the single test run's report and predictions (`test-run` first); they are drawn
after it and change nothing in it.

"On alert in an hour" means any alert during that hour, as the Active label counts it (Q7).

Usage: uv run python scripts/make_figures.py [explore|results]   (default: both)
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from airalert.backtest import PERIODS, TRAIN_START  # noqa: E402
from airalert.features import HOUR, KYIV, UnitTimeline, _seconds, build_grid  # noqa: E402
from airalert.labels import INTERIM_DIR  # noqa: E402
from airalert.settings import ROOT  # noqa: E402
from airalert.units import forecast_unit_ids  # noqa: E402

OUT = ROOT / "docs" / "figures"
HISTORY_START = pd.Timestamp("2023-01-01", tz=KYIV)  # labels start here (ADR-0006)
END = PERIODS["validation"][1]  # nothing from the test period

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, YELLOW, RED, NEUTRAL = "#2a78d6", "#eda100", "#e34948", "#a9a8a3"
LIGHT_BLUE = "#86b6ef"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def hourly_active(periods: pd.DataFrame, start, end) -> pd.DataFrame:
    """Hour (UTC) × unit: 1 if the unit was on alert at any moment of that hour."""
    hours = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    edges = _seconds(hours.append(pd.DatetimeIndex([hours[-1] + HOUR])))
    columns = {}
    by_unit = dict(tuple(periods.groupby("unit_id")))
    for unit_id in forecast_unit_ids():
        group = by_unit.get(unit_id, periods.iloc[:0])
        columns[unit_id] = (np.diff(UnitTimeline(group.started_at, group.finished_at).alert_seconds_before(edges)) > 0).astype(np.int8)
    return pd.DataFrame(columns, index=hours)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def share_by_month(active: pd.DataFrame) -> None:
    monthly = active.mean(axis=1).groupby(active.index.tz_convert(KYIV).strftime("%Y-%m")).mean()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = pd.to_datetime(monthly.index)
    ax.plot(x, monthly.to_numpy() * 100, color=BLUE, linewidth=2)
    ax.scatter(x[-1:], monthly.to_numpy()[-1:] * 100, color=BLUE, s=36, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"{monthly.iloc[-1]:.0%}", (x[-1], monthly.iloc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK)
    ax.set_title("Share of raion-hours with an air raid alert, by month")
    ax.set_ylabel("% of raion-hours")
    ax.set_ylim(0, max(60, monthly.max() * 110))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0, -0.2, "Kyiv months, 2023-01 to 2026-09-13 (the test period is left out). Source: @UkraineAlarmSignal.",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "alert_share_by_month.png")


def hour_by_weekday(active: pd.DataFrame) -> None:
    window = active[active.index >= TRAIN_START]
    local = window.index.tz_convert(KYIV)
    table = window.mean(axis=1).groupby([local.weekday, local.hour]).mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(9, 3.2))
    image = ax.imshow(table.to_numpy(), cmap=BLUES, aspect="auto")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("hour of day (Kyiv time)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("% of raion-hours on alert", color=INK_2)
    bar.outline.set_visible(False)
    ax.set_title("Alerts by hour and weekday")
    ax.text(0, -0.32, f"Share of raion-hours on alert, {TRAIN_START:%Y-%m-%d} to 2026-09-13 (training and validation periods).",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "alert_share_hour_weekday.png")


def oblast_by_month(active: pd.DataFrame) -> None:
    window = active[active.index >= TRAIN_START]
    month = window.index.tz_convert(KYIV).strftime("%Y-%m")
    oblast = window.columns.str.split("/").str[0].str.replace(" область", "")
    table = window.T.groupby(oblast).mean().T.groupby(month).mean().T * 100
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(table.to_numpy(), cmap=BLUES, aspect="auto", vmin=0, vmax=100)
    ax.set_yticks(range(len(table)), table.index)
    ax.set_xticks(range(len(table.columns)), [m[2:] for m in table.columns])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("% of raion-hours on alert", color=INK_2)
    bar.outline.set_visible(False)
    ax.set_title("Alerts by oblast and month (sorted by overall share)")
    ax.text(0, -0.07, "Average over the oblast's raions; months in Kyiv time; 2026-09 runs to 09-13. Luhansk is excluded (occupied).",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "alert_share_oblast_month.png")


def neighbours_and_onset(periods: pd.DataFrame, data_end) -> None:
    """For raions calm at t: how often an alert starts in the next hour, by the share of neighbours on alert at t."""
    grid = build_grid(periods, TRAIN_START, END, data_end, forecast_unit_ids())
    calm = grid[(grid.on_alert == 0) & grid.label_h1.notna()]
    bins = pd.cut(calm.neighbour_on_alert_share, [-0.01, 0, 0.5, 0.999, 1], labels=["none", "under half", "half or more", "all"])
    rate = calm.groupby(bins, observed=True).label_h1.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(rate))
    ax.bar(x, rate["mean"] * 100, width=0.6, color=BLUE)
    for i, p in enumerate(rate["mean"]):
        ax.text(i, p * 100 + 1.5, f"{p:.0%}", ha="center", color=INK)
    ax.set_xticks(x, [f"{name}\n(n = {n:,})" for name, n in zip(rate.index, rate["size"])])
    ax.set_xlabel("neighbouring raions on alert at t")
    ax.set_ylabel("% with an alert in the next hour")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Calm raions: alert in the next hour vs. neighbours")
    ax.text(0, -0.4, f"Raion-hours calm at t, {TRAIN_START:%Y-%m-%d} to 2026-09-13. Neighbours share a border (OCHA boundaries).",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "neighbours_and_onset.png")


def levels_by_day(alerts: pd.DataFrame) -> None:
    start = PERIODS["validation"][0]
    window = alerts[(alerts.started_at >= start) & (alerts.started_at < END) & ~alerts.retracted & ~alerts.missed_start]
    day = window.started_at.dt.tz_convert(KYIV).dt.strftime("%m-%d")
    level = window.max_color.fillna("none")
    table = pd.crosstab(day, level).reindex(columns=["none", "yellow", "red"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for column, color, name in (("none", NEUTRAL, "no level"), ("yellow", YELLOW, "yellow"), ("red", RED, "red")):
        values = table[column].to_numpy()
        ax.bar(x, values, bottom=bottom, width=0.75, color=color, label=name, edgecolor=SURFACE, linewidth=1)
        bottom += values
    ax.set_xticks(x[::2], table.index[::2], rotation=0, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("alert starts (raion level)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, bottom.max() * 1.18)  # room for the legend above the bars
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    switch = table.index.get_loc(next(d for d in table.index if table.loc[d, ["yellow", "red"]].sum() > 0))
    ax.axvline(switch - 0.5, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(switch - 0.4, bottom.max() * 0.97, "levels begin", color=INK_2, fontsize=8, va="top")
    ax.set_title("The switch to yellow/red alert levels (validation period)", pad=24)
    ax.text(0, -0.17, "Alert starts per day (Kyiv time), coloured by the highest level the alert reached. 2026-08-17 to 2026-09-13.",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "alert_levels_by_day.png")


MODELS = {  # name in the test report: (label, colour), baselines first
    "persistence": ("Persistence", NEUTRAL), "historical_rate": ("Historical rate", YELLOW),
    "logistic": ("Logistic regression", LIGHT_BLUE), "lightgbm": ("LightGBM", BLUE),
}
TEST_NOTE = "Test period 2026-09-14 to 2026-10-11 (Kyiv time), weekly retraining, all raions."


def spread(values: dict, gap: float) -> dict:
    """Label positions at least `gap` apart, each as close to its value as possible (bottom-up)."""
    out, last = {}, -np.inf
    for key, value in sorted(values.items(), key=lambda kv: kv[1]):
        out[key] = last = max(value, last + gap)
    return out


def brier_by_horizon(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, subset, title in ((axes[0], "all", "All raion-hours"), (axes[1], "calm at t", "Raions calm at t")):
        part = metrics[(metrics.units == "all") & (metrics.subset == subset)]
        ends = {}
        for model, (label, color) in MODELS.items():
            line = part[part.model == model].sort_values("horizon")
            x = np.arange(len(line))
            ax.plot(x, line.brier, color=color, linewidth=2.6 if model == "lightgbm" else 1.8, marker="o", markersize=4)
            ends[model] = line.brier.iloc[-1]
        if ax is axes[1]:
            for model, y in spread(ends, gap=0.012).items():
                label, color = MODELS[model]
                ax.annotate(label, (x[-1], y), xytext=(8, 0), textcoords="offset points", va="center",
                            color=color if model != "persistence" else INK_2, fontsize=9)
        ax.set_xticks(x, line.horizon.astype(str))
        ax.set_xlabel("horizon h (hours ahead)")
        ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].set_ylim(0, None)
    fig.suptitle("Forecast error by horizon, test period", x=0.06, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.06, -0.04, TEST_NOTE, color=INK_2, fontsize=8)
    save(fig, "brier_by_horizon.png")


def calibration(predictions: pd.DataFrame) -> None:
    """LightGBM: share of hours with an alert against the forecast probability, in bins of 0.1."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2)
    part = predictions[predictions.model == "lightgbm"]
    for h, color in ((1, BLUE), (6, LIGHT_BLUE), (24, YELLOW)):
        rows = part[part.horizon == h]
        bins = pd.cut(rows.p, np.linspace(0, 1, 11), include_lowest=True)
        table = rows.groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        table = table[table.n >= 200]
        ax.plot(table.p, table.y, color=color, marker="o", markersize=4, linewidth=1.8, label=f"h = {h}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("share of hours with an alert")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("LightGBM calibration, test period")
    ax.text(0, -0.2, "Bins of 0.1 with at least 200 raion-hours. On the diagonal, forecasts mean what they say.",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "calibration.png")


def example_forecast(predictions: pd.DataFrame, unit: str = "м. Київ", start="2026-10-04", end="2026-10-08") -> None:
    """One unit over a few days: the hours with an alert, and LightGBM's forecasts for each hour made 1 and 6 hours
    before it starts."""
    start, end = pd.Timestamp(start, tz=KYIV), pd.Timestamp(end, tz=KYIV)
    part = predictions[(predictions.model == "lightgbm") & (predictions.unit_id == unit)].copy()
    part["target"] = part.t + (part.horizon - 1) * HOUR  # start of the forecast hour
    part = part[(part.target >= start) & (part.target < end)]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    truth = part[part.horizon == 1]
    for target in truth.loc[truth.y == 1, "target"]:
        ax.axvspan(target.tz_convert(KYIV), (target + HOUR).tz_convert(KYIV), color="#e8e7e3", linewidth=0)
    for h, color, label in ((6, LIGHT_BLUE, "forecast made 6 h before the hour"), (1, BLUE, "forecast made 1 h before the hour")):
        line = part[part.horizon == h].sort_values("target")
        ax.step(line.target.dt.tz_convert(KYIV), line.p, where="post", color=color, linewidth=1.6, label=label)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability of an alert")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(tz=KYIV))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b", tz=KYIV))
    ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[6, 12, 18], tz=KYIV))
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title(f"Example: Kyiv city, {start:%-d} to {end - HOUR:%-d %B %Y}", pad=26)
    ax.text(0, -0.2, "Grey: hours with any alert (some lasted only about 5 minutes). Each forecast is drawn over the hour "
            "it is for. Kyiv time; test period.",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "example_forecast_kyiv.png")


def results() -> None:
    metrics = pd.read_csv(ROOT / "reports" / "test" / "metrics.csv")
    predictions = pd.read_parquet(ROOT / "data" / "results" / "test_predictions.parquet")
    print(f"Writing {OUT.relative_to(ROOT)}/ (test-period results)")
    brier_by_horizon(metrics)
    calibration(predictions)
    example_forecast(predictions)


def explore() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    periods = pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_label_periods.parquet")
    alerts = pd.read_parquet(INTERIM_DIR / "UkraineAlarmSignal_alerts.parquet")
    data_end = periods.data_end.iloc[0]
    active = hourly_active(periods, HISTORY_START.tz_convert("UTC"), END.tz_convert("UTC"))
    print(f"Writing {OUT.relative_to(ROOT)}/ ({active.shape[1]} units, {len(active):,} hours, before {END:%Y-%m-%d})")
    share_by_month(active)
    hour_by_weekday(active)
    oblast_by_month(active)
    neighbours_and_onset(periods, data_end)
    levels_by_day(alerts)


def main(which: str = "both") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("explore", "both"):
        explore()
    if which in ("results", "both"):
        results()


if __name__ == "__main__":
    main(*sys.argv[1:2])
