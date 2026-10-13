"""Threat posts from `@kpszsu` (Air Force) and `@deraketaua` (volunteer monitoring) as dated events (Q12).

Each line of a post is read on its own, because one post often lists several threats and places, and an
all-clear for one region next to threats for others ("⚠️3 шахеди на Харківщині … 📡Одещина – чисто"). A line gets
a set of threat types, read from its words first and, for the Air Force channel, from its leading emoji as a
fallback (the channel uses 🚀 for guided bombs too, so words win):

    drones, jet_drones, recon, missiles, ballistic, guided_bombs, aviation, intercept, clear

`clear` replaces the other types of its line: the line says a threat is over. "Відбій" and "чисто" always mean
that; "немає", "не спостерігається" and "відсутні" only on a short line, since longer lines with them are
explanations or news ("в мене немає десятки адмінів"). Lines with a Telegram handle or link are channel signatures
or adverts ("🍑 @deraketaua | Полтава | Київ") and are skipped: they name places that are not threatened. Places
come from places.find_places. A line becomes one row per (type, unit it names) with scope "local" (a city,
settlement or raion) or "oblast" (a whole oblast or region), plus one "national" row per type and post that every
unit can see. Rule-based, no language model (Q12).

Availability: a post counts from its edit time if it was edited, else from its publication time (Q12). Both feature
channels edit almost every post within seconds, so this costs little, and a post edited much later is never used
in a form it did not yet have. Post ids are unique only within a channel: (channel, post_id) identifies a post.
"""

import json
import re

import pandas as pd

from airalert.places import find_places
from airalert.settings import RAW_DIR, ROOT

CHANNELS = ("kpszsu", "deraketaua")
TYPES = ("drones", "jet_drones", "recon", "missiles", "ballistic", "guided_bombs", "aviation", "intercept", "clear")
CLEAR = re.compile(r"відбій|чисто")
WEAK_CLEAR = re.compile(r"не спостеріга|немає|відсутн")  # an all-clear only on a short line
WEAK_CLEAR_MAX_CHARS = 150
SIGNATURE = re.compile(r"@\w|t\.me/")
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


def lines_of(text: str | None) -> list[str]:
    """The non-empty lines of a post, without signature and link lines."""
    return [line.strip() for line in (text or "").splitlines() if line.strip() and not SIGNATURE.search(line)]


def threat_types(text: str | None, channel: str = "kpszsu") -> frozenset[str]:
    """Threat types of one line."""
    lowered = (text or "").lower().strip()
    if not lowered:
        return frozenset()
    if CLEAR.search(lowered) or (WEAK_CLEAR.search(lowered) and len(lowered) <= WEAK_CLEAR_MAX_CHARS):
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

    A line that names a place but no threat ("Курсом на Чорноморськ.", "3 на Богодухів з півночі") takes the threat
    types of the last line that had some, in the same post or in the channel's previous posts up to 30 minutes
    older, unless that line was an all-clear. It only looks back, and it counts from when both its own post and the
    one it borrows from were available, since the borrowed types come from that post's latest text.
    """
    rows, last = [], None  # last typed line: (types, posted, available_at)
    for post in sorted(posts, key=lambda p: p["id"]):
        posted = pd.Timestamp(post["date"])
        available_at = pd.Timestamp(post.get("edit_date") or post["date"])
        for line in lines_of(post.get("text")):
            types, seen = threat_types(line, channel), available_at
            mentions = find_places(line)
            if types:
                last = (types, posted, available_at)
            elif mentions and last is not None and posted - last[1] <= FOLLOW_UP and "clear" not in last[0]:
                types, seen = last[0], max(available_at, last[2])
            for kind in types:
                rows.append((post["id"], channel, seen, kind, "national", None))
                for mention in mentions:
                    scope = "oblast" if mention.kind == "oblast" else "local"
                    rows.extend((post["id"], channel, seen, kind, scope, unit_id) for unit_id in mention.unit_ids)
    frame = pd.DataFrame(rows, columns=["post_id", "channel", "available_at", "type", "scope", "unit_id"])
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame.sort_values("available_at", kind="stable").drop_duplicates(["post_id", "type", "scope", "unit_id"])


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
