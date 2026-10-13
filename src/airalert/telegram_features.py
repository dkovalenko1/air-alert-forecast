"""Telegram threat-post features on the hourly grid (Q12, ADR-0001).

For each unit and forecast time t, threat events (threats.py) are counted if they became available strictly before
t and within a look-back window:

- local: posts naming the unit itself (a city, settlement or its raion), by threat group, last 15 / 60 / 180 min;
- oblast: posts naming the unit's whole oblast or region, same groups and windows;
- neighbour: posts naming a neighbouring unit (shared border, Q11), any threat, last 60 / 180 min;
- national: all posts, by threat group, last 60 min, and any threat in the last 15 min;
- minutes since the last post naming the unit (capped at a week).

TELEGRAM_FEATURES is the full set (experiment E4). TARGETED_FEATURES is a small set chosen for stated reasons
(experiment E5, DESIGN.md section 3): any threat naming the unit, its oblast or a neighbour in the last hour, minutes
since the unit was last named, and the nationwide drone and missile posts of the last 3 h and aviation posts of the
last 6 h.

Threat groups: drones (incl. jet and reconnaissance drones), missiles (incl. ballistic), guided bombs, aviation,
intercepts, all-clear. A post counts once per group, however many of its types fall in it.
`reference_counts` recomputes one count naively; tests check both agree and that later posts never change a feature.
"""

import numpy as np
import pandas as pd

from airalert.features import CAP_MINUTES, NEIGHBOURS, _seconds

GROUPS = {
    "drones": ("drones", "jet_drones", "recon"),
    "missiles": ("missiles", "ballistic"),
    "guided_bombs": ("guided_bombs",),
    "aviation": ("aviation",),
    "intercept": ("intercept",),
    "clear": ("clear",),
}
THREAT_GROUPS = [g for g in GROUPS if g != "clear"]
WINDOWS_MIN = (15, 60, 180)
NEIGHBOUR_WINDOWS_MIN = (60, 180)

TELEGRAM_FEATURES = (
    [f"tg_{scope}_{group}_{w}m" for scope in ("local", "oblast") for group in GROUPS for w in WINDOWS_MIN]
    + [f"tg_neighbour_threat_{w}m" for w in NEIGHBOUR_WINDOWS_MIN]
    + [f"tg_national_{group}_60m" for group in GROUPS] + ["tg_national_threat_15m", "tg_minutes_since_local_threat"]
)
NATIONAL_LONG = {"drones": 180, "missiles": 180, "aviation": 360}  # group: window in minutes (E5)
TARGETED_FEATURES = (
    ["tg_local_threat_60m", "tg_oblast_threat_60m", "tg_neighbour_threat_60m", "tg_minutes_since_local_threat"]
    + [f"tg_national_{group}_{w}m" for group, w in NATIONAL_LONG.items()]
)
ALL_TELEGRAM_FEATURES = TELEGRAM_FEATURES + [c for c in TARGETED_FEATURES if c not in TELEGRAM_FEATURES]
TELEGRAM_SETS = {"all": TELEGRAM_FEATURES, "targeted": TARGETED_FEATURES}


POST = ["channel", "post_id"]  # post ids are unique only within a channel


def _grouped(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (post, group, scope, unit), at the earliest time any of it was available: a post counts once
    per group."""
    group_of = {kind: group for group, kinds in GROUPS.items() for kind in kinds}
    out = events.assign(group=events["type"].map(group_of)).dropna(subset=["group"]).sort_values("available_at", kind="stable")
    return out.drop_duplicates([*POST, "group", "scope", "unit_id"])


def _counts(times: np.ndarray, t: np.ndarray, window_s: int) -> np.ndarray:
    """Events with t - window <= time < t, for every t (times sorted, in seconds)."""
    return np.searchsorted(times, t, side="left") - np.searchsorted(times, t - window_s, side="left")


def add_telegram_features(grid: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = _grouped(events)
    hours = np.sort(grid["t"].unique())
    t = _seconds(hours)
    units = pd.Index(sorted(grid["unit_id"].astype(str).unique()))
    t_idx = np.searchsorted(t, _seconds(grid["t"]))
    u_idx = units.get_indexer(grid["unit_id"].astype(str))
    columns = {}

    def per_unit(selected: pd.DataFrame, window_s: int) -> np.ndarray:
        """Hours × units matrix of counts."""
        wide = np.zeros((len(t), len(units)), dtype=np.float32)
        for unit, part in selected.groupby("unit_id"):
            j = units.get_indexer([unit])[0]
            if j >= 0:
                wide[:, j] = _counts(np.sort(_seconds(part["available_at"])), t, window_s)
        return wide

    for scope in ("local", "oblast"):
        in_scope = events[events.scope == scope]
        for group in GROUPS:
            selected = in_scope[in_scope.group == group]
            for w in WINDOWS_MIN:
                columns[f"tg_{scope}_{group}_{w}m"] = per_unit(selected, w * 60)[t_idx, u_idx]

    threats = events[events.group != "clear"].drop_duplicates([*POST, "scope", "unit_id"])  # once per post
    local_threats = threats[threats.scope == "local"]
    for scope in ("local", "oblast"):
        columns[f"tg_{scope}_threat_60m"] = per_unit(threats[threats.scope == scope], 3600)[t_idx, u_idx]
    pairs = pd.read_csv(NEIGHBOURS)
    pairs = pairs[pairs.unit_id.isin(units) & pairs.neighbour_id.isin(units)]
    adjacency = np.zeros((len(units), len(units)), dtype=np.float32)
    adjacency[units.get_indexer(pairs.unit_id), units.get_indexer(pairs.neighbour_id)] = 1
    for w in NEIGHBOUR_WINDOWS_MIN:
        columns[f"tg_neighbour_threat_{w}m"] = (per_unit(local_threats, w * 60) @ adjacency.T)[t_idx, u_idx]

    national = events[events.scope == "national"]
    for group in GROUPS:
        times = np.sort(_seconds(national.loc[national.group == group, "available_at"]))
        columns[f"tg_national_{group}_60m"] = _counts(times, t, 3600).astype(np.float32)[t_idx]
    for group, w in NATIONAL_LONG.items():
        times = np.sort(_seconds(national.loc[national.group == group, "available_at"]))
        columns[f"tg_national_{group}_{w}m"] = _counts(times, t, w * 60).astype(np.float32)[t_idx]
    threat_posts = np.sort(_seconds(national[national.group != "clear"].drop_duplicates(POST)["available_at"]))
    columns["tg_national_threat_15m"] = _counts(threat_posts, t, 900).astype(np.float32)[t_idx]

    since = np.full((len(t), len(units)), CAP_MINUTES, dtype=np.float32)
    for unit, part in local_threats.groupby("unit_id"):
        j = units.get_indexer([unit])[0]
        if j < 0:  # a unit we do not forecast (e.g. occupied Luhansk raions)
            continue
        times = np.sort(_seconds(part["available_at"]))
        last = np.searchsorted(times, t, side="left") - 1  # the last post strictly before t
        since[:, j] = np.where(last >= 0, np.minimum((t - times[np.maximum(last, 0)]) / 60, CAP_MINUTES), CAP_MINUTES)
    columns["tg_minutes_since_local_threat"] = since[t_idx, u_idx]
    return pd.concat([grid.reset_index(drop=True), pd.DataFrame(columns)], axis=1)


def reference_counts(events: pd.DataFrame, unit_id: str, t: pd.Timestamp, scope: str, group: str, window_min: int) -> int:
    """One count, computed naively from the events available before t. `group="threat"`: any group but clear, each
    post from the first time it showed one; `scope="national"` ignores the unit."""
    e = _grouped(events)  # sorted by availability
    e = e[(e.scope == scope) & (True if scope == "national" else e.unit_id == unit_id)]
    e = e[e.group != "clear"].drop_duplicates(POST) if group == "threat" else e[e.group == group]
    known = e[(e.available_at < t) & (e.available_at >= t - pd.Timedelta(minutes=window_min))]
    return len(known[POST].drop_duplicates())
