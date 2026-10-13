# Air alert forecasting

Forecasting air raid alerts in Ukraine using only information that was public before the forecast is made.

## Language

### Alerts

**Alert**:
A declared period during which a location is under an air raid warning. It has a location, a start and an end.
_Avoid_: siren, alarm

**Location level**:
The administrative level an alert is declared for: oblast, raion or hromada. Since about autumn 2025, most alerts are declared per raion.

**Alert source**:
A record of alerts produced by one pipeline: official, volunteer or alerts.in.ua. Different sources define alerts differently, so they are not interchangeable.
_Avoid_: dataset (when the source is meant)

**Permanent alert**:
An alert that has not ended since 2022, in Luhansk oblast and Crimea. These locations are excluded from forecasting.

### Forecasts

**Unit level**:
The level forecasts are made for: oblast or raion.

**Unit**:
One specific oblast or raion that receives forecasts.
_Avoid_: region, area

**Forecast time (t)**:
The moment a forecast is issued. Only information available strictly before t may be used.
_Avoid_: now, issue date

**Training cutoff**:
The latest moment whose data a trained model has seen. It is never later than the forecast times the model is used for.

**Step**:
The length of one slot, e.g. 15 minutes, 1 hour or 1 day.

**Slot**:
A clock-aligned time interval one step long.

**Horizon (h)**:
How many steps ahead of t a slot lies. h = 1 is the slot that starts at t.
_Avoid_: lead time

**Target**:
What is predicted for a unit and slot. It is one of Active, Onset, Minutes or Time to alert.

**Active**:
The unit is under alert at some moment during the slot.

**Onset**:
A new alert starts in the unit during the slot.

**Minutes**:
How many minutes of the slot the unit is under alert.

**Time to alert**:
The number of slots until the next onset in the unit.
_Avoid_: time to event

**Forecast spec**:
A combination of target, unit level, step and horizons that defines one forecasting task.
_Avoid_: config, setup, configuration

**Main spec**:
The forecast spec that is analysed in depth in the report.

### Information and evaluation

**Availability time**:
The moment a piece of information became public, and from which forecasts may use it.
_Avoid_: timestamp

**As-of view**:
Everything that was known strictly before a given t. It is the only data a forecast at t may use.

**Calm at t / On alert at t**:
Whether the unit was under alert at the forecast time. Results are reported separately for the two groups: for a calm unit, an alert in the target hour has to start first, so the calm group shows how well the forecast anticipates new alerts. It is not an Onset target: an alert that started before the target hour also counts.

**Baseline**:
A simple rule that a model must beat. Persistence ("same state as at t"); Historical rate (the share of the last 4 weeks' slots with an alert at the same hour and weekday); Recent rate (the unit's share of slots with an alert in the last 4 weeks); Recent rate by state (the same, separately for slots when the unit was calm or on alert at t).

**Backtest**:
Repeating train-then-forecast over a sequence of training cutoffs, then scoring the forecasts against what actually happened.
_Avoid_: cross-validation

**Threat post**:
A Telegram post reporting a threat that has been launched or is in flight (drones, missiles, aircraft). It is distinct from posts that only announce alerts.
