# air-alert-forecast

Forecasting air raid alerts in Ukraine from information available before the forecast is made.

Work in progress. See [docs/DESIGN.md](docs/DESIGN.md) for the product overview, decisions and open questions, and [CONTEXT.md](CONTEXT.md) for the terms used.

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
2. Copy `.env.example` to `.env` and fill in:
   - an [alerts.in.ua API token](https://alerts.in.ua/api-request);
   - Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps).
3. Log in to Telegram once, in your own terminal:

   ```
   uv run python -m airalert telegram-login
   ```

   Telegram asks for your phone number and sends a code. The session is saved to `airalert.session`, which is git-ignored.

## Data

```
uv run python -m airalert download-telegram   # all channels; rerun to resume or to fetch new messages
uv run python scripts/fetch_alerts_in_ua.py   # alerts.in.ua history for the last month
```

Raw data goes to `data/raw/` and is not committed.
