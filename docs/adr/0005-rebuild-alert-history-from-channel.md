---
status: accepted
---

# The alert history is rebuilt from the `@air_alert_ua` messages, not taken from the published CSV

The published official CSV has two problems, and the CSV alone cannot undo them because it keeps no message type:
- Its parser treats every 🔴 message as an alert start, so artillery, urban-fighting and kamikaze-drone warnings become "air raid alerts".
- A repeated 🔴 message for a location already on alert overwrites the alert's start time.

We therefore download the channel's messages ourselves through Telethon and rebuild the alerts, keeping each message's type.

Parsing rules:
- **Messages are classified by wording, not emoji.**
- **Only "Повітряна тривога" → "Відбій тривоги" periods are air raid alerts.**
- **A repeated start keeps the original start time.**
- **Escalations never change an alert's start or end.**
- **A missing all-clear leaves an unknown period.** It is not trained on or scored, and no end time is invented.
- **An escalation with no open alert opens a flagged, implied alert.**
- **Unrecognised messages are logged and counted, never treated as alerts.**

## Considered options

- **Filter the CSV** (raion- and oblast-level rows only, plus hand-checked city zones): cheaper, but it keeps the overwritten start times.

## Consequences

- Telegram API credentials are needed before any alert labels can be built.
- The parser is checked against the CSV on rows where both should agree. The differences show how many CSV rows the two problems affected.
