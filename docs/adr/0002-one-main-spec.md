---
status: accepted
---

# Code accepts any forecast spec; only one main spec is analysed in depth

The code accepts any forecast spec: target × unit level × step × horizons. Only the **main spec** gets the full analysis: baselines, ablations, calibration, error analysis and write-up. A secondary spec answers one specific question: raion × 1 day × Minutes, for day-ahead planning. The oblast spec was dropped on 2026-10-12. Any other spec appears only in an automatically generated summary table.

**Update 2026-10-13:** only the main spec (raion × 1 h × Active) was built. The code is organised by spec, but the
secondary spec and the summary table of other specs were not implemented.

Analysing every combination in depth is not worth it, for three reasons:
- Checking and explaining results is the real cost: thousands of numbers across 24 specs × 24 horizons × several metrics.
- Compute: each spec is a full backtest.
- The selection trap: picking the best of many specs by its test-period score quietly tunes to the test data, so the reported score is optimistic.

Proposed main spec: **raion × 1 h × Active, horizons 1–24**, scored separately for units calm at t and on alert at t. It depends on open question Q2 in [DESIGN.md](../DESIGN.md).

## Considered options

- **One hard-coded task:** simpler, but the product should answer different questions (next hour, tomorrow, how many minutes).
- **Analysing the full grid:** too costly, and it invites the selection trap.
