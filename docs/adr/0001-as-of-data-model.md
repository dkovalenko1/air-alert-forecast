---
status: accepted
---

# Every record carries an availability time; features read data only through one as-of view

Leakage is the task's one hard rule. Every record in the canonical tables therefore carries an **availability time**:
- an alert's start is known from its start time;
- its end is known only from its end time;
- a Telegram post is known from its post time, or from its edit time if it was edited.

Features may read data only through a single as-of function. It returns what was known strictly before t, and it clips alerts still in progress at t, because their end is not yet known.

We chose this over building features ad hoc with shifted rolling windows. That approach spreads the leakage rule across dozens of places. A single choke point can be audited, and it can be tested by appending fake future data and checking that no feature changes.

## Consequences

- The feature pipeline is slower and more rigid than plain pandas rolling windows.
- Raw data is stored exactly as fetched and never edited. Everything downstream is rebuilt from it.
