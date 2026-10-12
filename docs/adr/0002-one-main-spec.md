---
status: accepted
---

# Code accepts any forecast spec; only one main spec is analysed in depth

The code accepts any forecast spec: target × unit level × step × horizons. Only the **main spec** gets the full analysis: baselines, ablations, calibration, error analysis and write-up. A few secondary specs answer one question each. Any other spec appears only in an automatically generated summary table.

Analysing every combination in depth is not worth it, for three reasons:
- Checking and explaining results is the real cost: thousands of numbers across 24 specs × 24 horizons × several metrics.
- Compute: each spec is a full backtest.
- The selection trap: picking the best of many specs by its test-period score quietly tunes to the test data, so the reported score is optimistic.

Proposed main spec: **raion × 1 h × Active, horizons 1–24**, scored separately for units calm at t and on alert at t. It depends on open question Q2 in [DESIGN.md](../DESIGN.md).

## Considered options

- **One hard-coded task:** simpler, but the product should answer different questions (next hour, tomorrow, how many minutes).
- **Analysing the full grid:** too costly, and it invites the selection trap.
