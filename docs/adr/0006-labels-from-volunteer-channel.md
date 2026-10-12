---
status: accepted
---

# Alert labels come from `@UkraineAlarmSignal` from 2023; the official channel is the validation reference

The official channel `@air_alert_ua` stopped posting on 2026-09-07, and it never carried the yellow/red levels. `@UkraineAlarmSignal` (eTryvoga) is live. It has posted alerts since 2022-02-25, at raion level since 2025 and with levels since September 2026.

In spot checks (12-hour windows in 2023, 2024 and 2025, plus 6 hours on 2026-09-06), the channel contained practically every official alert start and end, with a median delay under 45 s.

So labels come from this one channel for the whole history from 2023-01-01. One source and one parser cover training and test, with no switch at the boundary (ADR-0003).

Supporting roles:
- **`@air_alert_ua`:** rebuilt with the same parsing rules (ADR-0005) and used as the validation reference up to 2026-09-07.
- **alerts.in.ua:** an independent cross-check for the latest month.

## Considered options

- **Official channel until 2026-09-07, then this channel or alerts.in.ua:** the alert source would change exactly at the train/test boundary.
- **Including 2022:** the channel's 2022 messages miss some official alerts and sometimes name only a city, so 2022 is left out.

## Consequences

- The parser must handle the channel's changing formats: oblast-level messages (2023–2024), raion-level ones (2025 →), level messages, and messages listing several raions.
- It must also drop the channel's own city-level warnings (🛸 drones, 💣 guided bombs, 💥 shelling, ⚠️ media reports).
- The spot checks must be repeated over whole months before the labels are trusted.
- Labels depend on a volunteer project. If the channel stops, alerts.in.ua is the fallback.
