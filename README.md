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

## No leakage

Nothing a forecast for time t uses, and nothing its model is trained on, comes from after t
([ADR-0001](docs/adr/0001-as-of-data-model.md)). The main rules:

- **Features use only what was known before t.** An alert still on at t counts as ongoing, however long it later
  lasted. Tests compare the fast feature code with a naive "what was known at t" computation, and check that
  rewriting everything after t changes no feature at t.
- **Training labels must be known by the cutoff:** a model retrained at cutoff C only uses rows with t + h ≤ C.
- **Corrections are not known early.** An alert later retracted as a mistake counts as on alert until the
  retraction was posted; only the labels drop it ([ADR-0007](docs/adr/0007-published-vs-corrected-history.md)).
- **The unit list is fixed in advance,** so a raion's first alert never changes country or neighbour features before it.
- **Edited messages.** Telegram returns only a message's latest text, dated at its original posting time. In the
  label channel every message has been edited, mostly in two channel-wide re-edits (2024-01-30 and 2025-06-27)
  and, since July 2025, automatically about 20 seconds after posting. Sampled late edits keep the channel's fixed
  wording, so edits are treated as not changing an alert's kind or place. The month-by-month comparison with the
  independent official channel would catch it if they did. Telegram feature posts edited at or after t are not
  used at t (ADR-0007).

## Sources and attribution

- Alert labels: Telegram channel [`@UkraineAlarmSignal`](https://t.me/UkraineAlarmSignal) (єТривога), checked against the official channel [`@air_alert_ua`](https://t.me/air_alert_ua) and [alerts.in.ua](https://alerts.in.ua).
- Raion list: [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) by Vadym Klymenko (MIT).
- Raion borders for the neighbour table: OCHA, [Ukraine – Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-ukr), HDX, CC BY-IGO. Only the derived neighbour pairs (`src/airalert/reference/neighbours.csv`) are in this repo.
