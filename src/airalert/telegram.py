"""Download Telegram channel history through the official API (ADR-0004).

Messages are stored as fetched (ADR-0001), one JSON object per line, in
data/raw/telegram/<channel>/<YYYY-MM>.jsonl. A rerun continues after the last saved message,
so the same command both resumes an interrupted download and fetches new messages later.
"""

import json
import pathlib
import sys
import time

from telethon.sync import TelegramClient
from telethon.tl.types import MessageService

from airalert.settings import RAW_DIR, TELEGRAM_SESSION, secret

# Label source first (ADR-0006), then the validation reference, then the feature channels (Q12).
CHANNELS = ("UkraineAlarmSignal", "air_alert_ua", "kpszsu", "deraketaua")
ASSUMED_RATE = 80  # messages per second with Telethon's 1 s pause per 100 messages; replaced by the measured rate


def _client() -> TelegramClient:
    client = TelegramClient(str(TELEGRAM_SESSION), int(secret("TELEGRAM_API_ID")), secret("TELEGRAM_API_HASH"))
    client.flood_sleep_threshold = 3600  # wait out Telegram's rate limits instead of failing
    return client


def login() -> None:
    """Log in once; Telegram asks for the phone number and sends a code. Run in your own terminal."""
    with _client():  # `with` starts the client, which prompts for the phone number and code on first use
        print(f"Logged in. Session saved to {TELEGRAM_SESSION}.session (git-ignored).")


def _record(message) -> dict:
    return {
        "id": message.id,
        "date": message.date.isoformat(),
        "edit_date": message.edit_date.isoformat() if getattr(message, "edit_date", None) else None,
        "text": getattr(message, "message", None),
        "service": type(message.action).__name__ if isinstance(message, MessageService) else None,
        "media": type(message.media).__name__ if getattr(message, "media", None) else None,
        "grouped_id": getattr(message, "grouped_id", None),
    }


def _last_saved_id(channel_dir: pathlib.Path) -> int:
    """Highest saved message id. Drops a half-written last line left by an interrupted run."""
    files = sorted(channel_dir.glob("*.jsonl"))
    if not files:
        return 0
    # Split on "\n" only: str.splitlines() would also split at characters such as U+2028 inside a message's text.
    lines = files[-1].read_text(encoding="utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()  # the file ends with a newline
    dropped = False
    while lines:
        try:
            last_id = json.loads(lines[-1])["id"]
            break
        except (json.JSONDecodeError, KeyError):
            lines.pop()
            dropped = True
    else:
        files[-1].unlink()
        return _last_saved_id(channel_dir)
    if dropped:  # rewrite only to remove a half-written last line; otherwise leave the raw file untouched
        files[-1].write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return last_id


def download(channel: str) -> None:
    channel_dir = RAW_DIR / "telegram" / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    start_id = _last_saved_id(channel_dir)

    client = _client()
    client.connect()
    if not client.is_user_authorized():
        sys.exit("Not logged in to Telegram: run `uv run python -m airalert telegram-login` in your terminal first.")
    try:
        latest_id = client.get_messages(channel, limit=1)[0].id
        expected = max(latest_id - start_id, 0)
        print(f"{channel}: messages {start_id + 1}..{latest_id}, up to {expected:,} new, "
              f"about {expected / ASSUMED_RATE / 60:.0f} min", flush=True)

        out, out_month, fetched, started = None, None, 0, time.monotonic()
        try:
            for message in client.iter_messages(channel, reverse=True, min_id=start_id, wait_time=1):
                month = message.date.strftime("%Y-%m")
                if month != out_month:
                    if out:
                        out.close()
                    out, out_month = (channel_dir / f"{month}.jsonl").open("a", encoding="utf-8"), month
                out.write(json.dumps(_record(message), ensure_ascii=False) + "\n")
                fetched += 1
                if fetched % 10_000 == 0:
                    out.flush()
                    rate = fetched / (time.monotonic() - started)
                    left = max(expected - fetched, 0) / rate / 60
                    print(f"  {fetched:,} saved (up to {message.date:%Y-%m-%d}), {rate:.0f} msg/s, about {left:.0f} min left",
                          flush=True)
        finally:
            if out:
                out.close()
        print(f"{channel}: {fetched:,} messages saved to {channel_dir}", flush=True)
    finally:
        client.disconnect()
