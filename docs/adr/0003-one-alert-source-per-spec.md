---
status: accepted
---

# Each forecast spec uses one alert definition from training through test

The available alert sources define alerts differently. Official and volunteer data agree on only 80.6% of oblast-hours (June–September 2026). If the source changed at the train/test boundary, the model would be scored against a different target from the one it learned.

Each spec therefore uses a single definition end to end:
- The official record ends on 2026-09-07. It is extended with alerts.in.ua history only after a consistency check (open question Q2 in [DESIGN.md](../DESIGN.md)).
- No raion-level source covers 2026-09-07 to about 2026-09-12. That period is marked missing and is not trained on or scored.

## Considered options

- **Volunteer data after 2026-09-07:** it is current, but it has a different alert definition and is oblast-level only.
- **UkraineAlarm API:** it might cover the gap, but key requests are answered slowly and there is no time to wait.
