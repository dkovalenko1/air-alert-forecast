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
- **Training labels must be known by the cutoff:** a model retrained at cutoff C only uses rows with t + h ≤ C,
  each with its label as known at C (a correction posted later is not applied yet).
- **Validation never sees the test period:** a forecast is scored only if its target hour lies inside its own period.
- **Corrections are not known early.** An alert later retracted as a mistake counts as on alert until the
  retraction was posted; only the labels drop it ([ADR-0007](docs/adr/0007-published-vs-corrected-history.md)).
- **The unit list is fixed in advance,** so a raion's first alert never changes country or neighbour features before it.
- **Edited messages.** Telegram returns only a message's latest text, dated at its original posting time. In the
  label channel every message has been edited, mostly in two channel-wide re-edits (2024-01-30 and 2025-06-27)
  and, since July 2025, automatically about 20 seconds after posting. Sampled late edits keep the channel's fixed
  wording, so edits are treated as not changing an alert's kind or place. This is an evidence-backed assumption;
  the month-by-month comparison with the independent official channel is an approximate cross-check, still to run.
  For Telegram feature posts, the planned rule is stricter: a post edited at or after t is not used at t (ADR-0007).

## Label quality

Labels come from the volunteer channel `@UkraineAlarmSignal` (єТривога), the only source that covers both
training and the latest weeks ([ADR-0006](docs/adr/0006-labels-from-volunteer-channel.md)). Before building on
it, we compared it with the official channel `@air_alert_ua` month by month, against a pass mark fixed in
advance: in every month, at least 98% of official alert starts and all-clears appear in our labels within 5
minutes, and vice versa. `uv run python -m airalert check-labels` writes the tables to
[`reports/label_quality/`](reports/label_quality/).

**Result: the pass mark is not met.** 33 of 176 month × kind × direction rows are below 98% (lowest 96.2%; 18
months, mostly 2023). The causes we found:

- **The channel misses some all-clears.** The alert then seems to go on for hours or days (all of Lviv oblast
  "on alert" from 3 to 15 April 2026). A stretch with no news for longer than the oblast's limit, or ended by a new
  start more than 20 minutes later, is marked **unknown**: it is neither trained on nor scored. The limits were
  calibrated on the official channel before the validation period (`scripts/calibrate_missing_all_clear.py`).
- **The channel misses some starts:** it posts an all-clear for an alert it never announced. The 5 hours before
  such an all-clear (or since the place's previous message) are marked unknown; 95% of those official alerts
  started within that window.
- **The official record has faults too:** a 4-day outage (2026-08-29 to 09-02, left out of the comparison) and
  some broken alerts of its own (e.g. 135 hours across Lviv and Chernivtsi raions in November 2025).
- A parser bug (messages with several sections, since July 2026) was found this way and fixed.

Missing messages cannot be recovered, so the message-level check still fails. **At the level the model uses,
"was this raion on alert in this hour", the labels agree with the official record on 99.5–99.7% of known
raion-hours per year (2023–2026), and on at least 97.3% in every month** (the lowest is November 2025, the month
of the official channel's own broken alert). Unknown hours: 0.3% of raion-hours in 2023, 3.6% in 2026; 1.2% of the
validation period and 0.4% of the test period. After the official channel stopped (2026-09-07), the test period
is checked against alerts.in.ua instead: every month passes (98.4–99.96%), with 99.8% hour-level agreement.

## Sources and attribution

- Alert labels: Telegram channel [`@UkraineAlarmSignal`](https://t.me/UkraineAlarmSignal) (єТривога), checked against the official channel [`@air_alert_ua`](https://t.me/air_alert_ua) and [alerts.in.ua](https://alerts.in.ua).
- Raion list: [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) by Vadym Klymenko (MIT).
- Raion borders for the neighbour table: OCHA, [Ukraine – Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-ukr), HDX, CC BY-IGO. Only the derived neighbour pairs (`src/airalert/reference/neighbours.csv`) are in this repo.
