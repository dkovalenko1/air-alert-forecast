"""Threat posts from `@kpszsu` (Air Force) and `@deraketaua` (volunteer monitoring) as dated events (Q12).

Each post gets a set of threat types, read from its words first and, for the Air Force channel, from its leading
emoji as a fallback (the channel uses 🚀 for guided bombs too, so words win):

    drones, jet_drones, recon, missiles, ballistic, guided_bombs, aviation, intercept, clear

`clear` ("відбій", "чисто", "немає", "не спостерігається") replaces the other types: such a post says a threat is
over. Places come from places.find_places. A post becomes one row per (type, unit it names) with scope "local"
(a city, settlement or raion) or "oblast" (a whole oblast or region), plus one "national" row per type that every
unit can see. Rule-based, no language model (Q12).

Availability: a post counts from its edit time if it was edited, else from its publication time (Q12). Both feature
channels edit almost every post within seconds, so this costs little, and a post edited much later is never used
in a form it did not yet have.
"""

import json
import re

import pandas as pd

from airalert.places import find_places
from airalert.settings import RAW_DIR, ROOT

CHANNELS = ("kpszsu", "deraketaua")
TYPES = ("drones", "jet_drones", "recon", "missiles", "ballistic", "guided_bombs", "aviation", "intercept", "clear")
CLEAR = re.compile(r"відбій|чисто|не спостеріга|немає|відсутн")
WORDS = [  # (pattern on the lowercased text, type)
    (r"реактивн", "jet_drones"),
    (r"розвідувальн", "recon"),
    (r"бпла|шахед|мопед|дрон|баражуюч", "drones"),
    (r"балістик|балістичн|іскандер|кинджал|високошвидкісн", "ballistic"),
    (r"\bкаб|авіабомб", "guided_bombs"),
    (r"ракет|крилат|калібр|циркон|\bх-\d", "missiles"),
    (r"авіаці|бомбардувальник|ракетоносі|ту-\d|міг-31", "aviation"),
    (r"збиття|збито|збил|робота ппо|вибух", "intercept"),
]
EMOJI = {"🛵": "drones", "🏍": "jet_drones", "🛸": "recon", "☄": "ballistic", "💣": "guided_bombs", "🛫": "aviation",
         "🚀": "missiles"}
EVENTS = ROOT / "data" / "interim" / "telegram_events.parquet"


def threat_types(text: str | None, channel: str = "kpszsu") -> frozenset[str]:
    lowered = (text or "").lower()
    if not lowered.strip():
        return frozenset()
    if CLEAR.search(lowered):
        return frozenset({"clear"})
    found = {kind for pattern, kind in WORDS if re.search(pattern, lowered)}
    if found & {"jet_drones", "recon"}:
        found.discard("drones")  # "реактивний / розвідувальний БпЛА": the more specific drone type
    if "ballistic" in found or "guided_bombs" in found:
        found.discard("missiles")  # "балістичні ракети", "КАБи": the more specific type
    if not found and channel == "kpszsu":
        lead = re.match(r"^\W+", text.strip())
        found = {kind for symbol, kind in EMOJI.items() if lead and symbol in lead.group(0)}
    return frozenset(found)


FOLLOW_UP = pd.Timedelta(minutes=30)


def events_from_posts(posts: list[dict], channel: str) -> pd.DataFrame:
    """Rows: post_id, channel, available_at, type, scope (local / oblast / national), unit_id (None if national).

    A follow-up post that names a place but no threat ("Курсом на Чорноморськ.") takes the threat types of the
    channel's previous typed post, if that one is at most 30 minutes older. It only looks back, so it cannot leak.
    """
    rows, last_types, last_time = [], frozenset(), None
    for post in sorted(posts, key=lambda p: p["id"]):
        types = threat_types(post.get("text"), channel)
        posted = pd.Timestamp(post["date"])
        mentions = find_places(post["text"]) if post.get("text") else []
        if types:
            last_types, last_time = types, posted
        elif mentions and last_time is not None and posted - last_time <= FOLLOW_UP and "clear" not in last_types:
            types = last_types
        if not types:
            continue
        available_at = pd.Timestamp(post.get("edit_date") or post["date"])
        for kind in types:
            rows.append((post["id"], channel, available_at, kind, "national", None))
            for mention in mentions:
                scope = "oblast" if mention.kind == "oblast" else "local"
                rows.extend((post["id"], channel, available_at, kind, scope, unit_id) for unit_id in mention.unit_ids)
    frame = pd.DataFrame(rows, columns=["post_id", "channel", "available_at", "type", "scope", "unit_id"])
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame.drop_duplicates(["post_id", "type", "scope", "unit_id"])


def build(since: str = "2025-01-01") -> None:
    """Write data/interim/telegram_events.parquet from the downloaded feature channels."""
    parts = []
    for channel in CHANNELS:
        posts = []
        for path in sorted((RAW_DIR / "telegram" / channel).glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                posts.extend(p for p in map(json.loads, f) if p["date"] >= since and p.get("text"))
        events = events_from_posts(posts, channel)
        typed = events.post_id.nunique()
        local = events[events.scope != "national"].post_id.nunique()
        print(f"{channel}: {len(posts):,} posts since {since}, {typed:,} with a threat type ({typed / max(len(posts), 1):.0%}), "
              f"{local:,} of them name a place; {len(events):,} events")
        parts.append(events)
    events = pd.concat(parts, ignore_index=True)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS, index=False)
    print(events[events.scope == "national"].groupby(["channel", "type"]).size().unstack(fill_value=0).to_string())
