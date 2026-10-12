"""Exploratory figures for the README and notebook (docs/figures/*.png).

Built from the label periods (`build-labels UkraineAlarmSignal` first). Only data before the test period is used
(up to the end of validation, 2026-09-13): these charts inform modelling choices, so the test weeks stay unseen.

"On alert in an hour" means any alert during that hour, as the Active label counts it (Q7).

Usage: uv run python scripts/make_figures.py
"""

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
    window = alerts[(alerts.started_at >= start) & (alerts.started_at < END) & ~alerts.retracted]
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


def main() -> None:
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


if __name__ == "__main__":
    main()
