---
status: accepted
---

# Telegram posts are downloaded through the official API (Telethon)

Posts are downloaded with Telethon, using the user's own Telegram API credentials.

## Considered options

- **Public web preview (`t.me/s/<channel>`):** needs no login. It is slower (20 posts per page) and does not expose edit times. The leakage check needs edit times: a post edited after t must not be used in its edited form.
- **Telegram Desktop "Export chat history":** gives about the same content. But it is manual, cannot be rerun by graders, and cannot fetch only new posts before the final evaluation.

## Consequences

- The user needs a Telegram account and `api_id`/`api_hash` credentials, kept in `.env`.
- The first login happens in the user's terminal. The session file is git-ignored.
- **Applies to every channel** (confirmed 2026-10-12): `@UkraineAlarmSignal`, `@air_alert_ua`, `@kpszsu`, `@deraketaua`.
- **Desktop exports:** a manual Desktop export may be used to explore data, but the pipeline never reads one.
