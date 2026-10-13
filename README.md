# air-alert-forecast

Forecasting air raid alerts in Ukraine from information available before the forecast is made.

For every raion (118 raions and Kyiv city) and every whole hour t, the model gives the probability that the raion
is under an air raid alert at any moment of the hour starting h − 1 hours after t, for h = 1, 2, 3, 6, 12 and 24.
It uses only what was published before t. It is meant for planning ("how likely is an alert here this evening?"),
not as a warning system: always follow official alerts.

Design, decisions and the full evaluation protocol: [docs/DESIGN.md](docs/DESIGN.md) and [docs/adr/](docs/adr/).
Terms: [CONTEXT.md](CONTEXT.md). The conversation with the AI assistant that built this:
[docs/ai-conversation.md](docs/ai-conversation.md).

## Results

**Test period: 14 September – 11 October 2026** (Kyiv time), the latest four weeks of data. Models were retrained
every week on everything known at the start of that week. All model and setting choices were made on the
validation period before it (17 August – 13 September), and no choice was changed because of test scores.

The test period was run three times, and every run is in the git history. The first (commit b30da01) was the plan.
An external review then found bugs in the Telegram features and two label edge cases, so it was rerun with the same
configuration (2ca5ac6); a later review found one more parser case (all-clears that start with a time), so it was
run once more (this version, [`reports/test/`](reports/test/)). The LightGBM score stayed at 0.118 each time. Two
more baselines were added after the first run, at reviewers' suggestions (Recent rate and Recent rate by state);
they were not used to choose anything.

Brier score, lower is better (the best model in each column is bold):

| Brier score | h = 1 | 2 | 3 | 6 | 12 | 24 | mean |
|---|---|---|---|---|---|---|---|
| Persistence ("same as now") | 0.122 | 0.209 | 0.222 | 0.238 | 0.247 | 0.251 | 0.215 |
| Historical rate (same hour and weekday, last 4 weeks) | 0.157 | 0.157 | 0.157 | 0.157 | 0.158 | 0.159 | 0.158 |
| Recent rate (the raion's alert share, last 4 weeks) | 0.132 | 0.132 | 0.132 | 0.132 | 0.133 | 0.133 | 0.132 |
| Recent rate by state (the same, separately for raions on alert / calm at t) | 0.080 | 0.124 | 0.128 | 0.131 | **0.132** | **0.133** | 0.121 |
| Logistic regression | 0.078 | 0.121 | 0.125 | 0.131 | 0.135 | 0.139 | 0.122 |
| LightGBM | **0.076** | **0.117** | **0.122** | **0.127** | 0.132 | 0.134 | **0.118** |

![Forecast error by horizon](docs/figures/brier_by_horizon.png)

**What a forecast means.** The forecast "h = 1" is for the hour that starts at t, so it is not an alert-start
forecast: 30% of those rows are already on alert at t, and the model scores almost perfectly on them. The "calm at t"
rows are the hard part, and the figures and tables below give both.

- **LightGBM is clearly best only in the next few hours.** It beats every other model at 1, 2, 3 and 6 hours. Against
  the strongest simple baseline, Recent rate by state (two numbers per raion: how often it had an alert in the last
  four weeks, when it was on alert at t and when it was calm), its lead is 0.0044 at 1 hour (95% interval
  0.0031–0.0058) and 0.0032 at 6 hours (0.0011–0.0052). At 12 and 24 hours the two cannot be told apart: LightGBM is
  behind by 0.0009 and 0.0013, with intervals that include zero
  ([`differences_by_horizon.csv`](reports/test/differences_by_horizon.csv); intervals from resampling days). On
  validation LightGBM was ahead at 24 hours (0.123 against 0.129), so this showed up only on the test period. In
  short: the model knows what happens in the next few hours; a day ahead, it knows little more than how often the
  raion has been on alert lately.
- **On average over the six horizons** LightGBM is best: 0.1181, against 0.1213 for Recent rate by state (lead
  0.0032, 95% interval 0.0014–0.0049), 0.1217 for logistic regression (0.0036, 0.0021–0.0052), 0.1324 for Recent rate
  and 0.1576 for Historical rate ([`differences.csv`](reports/test/differences.csv)). It has the lowest score in each
  of the four test weeks among the models in the table.
- **Raions calm at t** (the hard part; an alert in the target hour has to start first, or has started since):
  LightGBM scores 0.108 one hour ahead, against 0.115 for Recent rate by state, 0.128 for Recent rate and 0.154 for
  Historical rate. Pooled ROC-AUC is 0.845 one hour ahead and 0.839 a day ahead, but **most of that is telling
  raions apart**: averaged within each raion, it is 0.66 one hour ahead and 0.54 a day ahead
  ([`within_units.csv`](reports/test/within_units.csv)). So a day ahead, the model is much better at saying which
  raions are risky than at saying when.
- **The probabilities are well calibrated overall** (below). One hour ahead, forecasts of 30–40% came true 35% of the
  time. A day ahead, mid-range forecasts run a little low: 30–40% came true 43% of the time, probably because the
  test period had more alerts than the weeks the model learnt from. Per raion, the average forecast is off by 0.04
  at 24 hours, and by up to 0.16 for a single raion (Mykolaiv raion: forecasts too low).
- **The test period was harder than validation** (LightGBM 0.118 against 0.109): alerts were on in 42% of
  raion-hours, against 38%.
- **17 frontline units** (Donetsk 8, Zaporizhzhia 5, Kharkiv 2, Dnipropetrovsk 1, Sumy 1) were on alert in at least 90% of
  validation hours. They are easy to forecast. Without them, every score rises (LightGBM 0.134, Recent rate by state
  0.138, Recent rate 0.151, Historical rate 0.180), and the ranking stays the same.

<p>
  <img src="docs/figures/calibration.png" width="42%" alt="Calibration of LightGBM on the test period">
</p>

![Example forecasts for Kyiv city](docs/figures/example_forecast_kyiv.png)

In the example, the forecast made as the hour starts (h = 1) jumps to 1 when an alert is already on at that moment.
Otherwise, its probability rises and falls with the activity around the city. On the morning of 4 October, several
Kyiv alerts lasted about five minutes each. Starting from a calm hour, no forecast can tell which hours will get one.

### Telegram threat posts did not help

Posts from the Air Force (`@kpszsu`) and a volunteer monitoring channel (`@deraketaua`) were turned into dated
threat events, using rules with no language model: drones, missiles, ballistic, guided bombs, aviation,
intercepts, all-clear; the places named; the time of the post or its last edit. They were tested twice on
validation, with the rule fixed in advance (keep only if the mean Brier score improves by at least 0.0005):

- **47 features** (E4): no change for logistic regression (0.11013 → 0.11014), and LightGBM got worse.
- **7 targeted features** (E5, designed after E4 failed): logistic regression improved by 0.0001, and LightGBM got
  worse.

An external review of the first version found two leaks and three extraction bugs in these features (e.g. post
ids clashing between the two channels, and an all-clear for one region erasing the threats listed for others).
The numbers here are after the fixes. Before them, Telegram looked slightly better, and the decisions were the same.

Neither set passed. On the test period, adding the 7 features changed LightGBM by +0.0005 (−0.0001 … +0.0011) and
logistic regression by +0.0001 (−0.0009 … +0.0011): no measurable effect either way. Our reading, not tested: a post naming a
raion mostly arrives when its neighbours are already on alert, which the model already sees. And a post gives
minutes of warning, while the forecast is by the hour.

### How the model was chosen (validation)

Each experiment's protocol was written down and committed before it ran
([DESIGN.md, section 3](docs/DESIGN.md#3-evaluation-protocol)). Every variant's metrics are in
[`reports/validation/experiments/`](reports/validation/experiments/).

| Experiment | Winner (mean Brier) | Runner-up or default |
|---|---|---|
| E1. Training start | 1 November 2025 (LightGBM 0.1105) | 2023-01-01 0.1123; 2025-08-01 0.1127 |
| E2. One shared model or one per raion | shared (0.1105) | per raion 0.1221 |
| E3. LightGBM settings | without raion identity (0.1088) | default 0.1105; four others 0.1101–0.1117 |
| E4. All Telegram features | without (0.1088) | with 0.1091 |
| E5. Targeted Telegram features | without (0.1088) | with 0.1097 |

The scores are LightGBM's. Logistic regression made the same choices in E1, E2, E4 and E5 (E3 was for LightGBM only).

**The final model:** LightGBM, one model per horizon shared by all raions, 300 trees with 63 leaves, trained from
1 November 2025. It doesn't know which raion it is forecasting. It sees only what is happening there and around it:
- the raion's alert state and recent history (on alert now, how long, time since the last all-clear, alert starts
  in the last 1/3/6/24 h, share of the last day and week on alert);
- the share of its neighbours, its oblast and the whole country on alert, and alert starts in the last hour among
  its neighbours and nationwide;
- whether the target hour had an alert a day and a week earlier;
- the target hour, weekday and month.

## Using it

Setup:

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
2. Copy `.env.example` to `.env` and fill in:
   - an [alerts.in.ua API token](https://alerts.in.ua/api-request) (only for the test-period label cross-check);
   - Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps).
3. Log in to Telegram once, in your own terminal:

   ```
   uv run python -m airalert telegram-login
   ```

   Telegram asks for your phone number and sends a code. The session is saved to `airalert.session`, which is git-ignored.

Forecast from the latest data:

```
uv run python -m airalert download-telegram   # all four channels; rerun to resume or to fetch new messages
uv run python -m airalert build-labels        # alert periods per raion from the channel messages
uv run python -m airalert build-telegram      # threat events from @kpszsu and @deraketaua
uv run python -m airalert train               # fit the chosen model on everything up to the last whole hour (~1 min)
uv run python -m airalert predict             # probabilities per raion and horizon (seconds; --at for an earlier hour)
```

An excerpt of real output:

```
Forecast time: 2026-10-13 00:00 Kyiv (data up to 2026-10-13 00:50); model: lightgbm, trained at 2026-10-13 00:00 Kyiv
Probability of an alert in the hour starting h - 1 hours from now:

                                              unit  on alert  h=1  h=2  h=3  h=6  h=12  h=24
                                           м. Київ      True 1.00 0.64 0.59 0.62  0.68  0.77
              Харківська область/Харківський район      True 1.00 0.88 0.83 0.86  0.90  0.92
       Дніпропетровська область/Дніпровський район     False 0.63 0.62 0.60 0.61  0.71  0.72
                    Одеська область/Одеський район     False 0.39 0.41 0.41 0.40  0.48  0.57
                Львівська область/Львівський район     False 0.01 0.02 0.01 0.02  0.03  0.08
```

`predict` refuses a forecast time earlier than the saved model's training cutoff, so a forecast never uses a model
that learnt from later data.

Reproduce the evaluation:

```
uv run python scripts/fetch_alerts_in_ua.py    # alerts.in.ua history, for the test-period label cross-check
uv run python -m airalert build-labels         # rebuild both label tables first: check-labels reads them
uv run python -m airalert check-labels         # label quality (below)
uv run python -m airalert backtest validation  # baselines and models on validation
uv run python -m airalert experiment training-start   # then per-unit, lightgbm, telegram, telegram-targeted
uv run python -m airalert test-run --out reports/test-rerun   # reproduce the test run (the committed reports/test is
                                                               # the reported one; test-run refuses to overwrite it)
uv run python -m airalert test-report          # intervals, weekly and per-raion scores from the saved predictions
uv run python scripts/replay_edit_times.py     # validation: cost of the pessimistic edited-messages assumption
uv run python scripts/unknown_hours_check.py   # validation: the model on hours with unknown labels
uv run python scripts/make_figures.py          # docs/figures/
uv run pytest                                  # includes the leakage tests
```

Raw data goes to `data/` and saved models to `models/`; neither is committed. Telegram returns each message's latest
text, so a later download of the same months can differ (edits, deleted messages).
[`reports/test/data_manifest.csv`](reports/test/data_manifest.csv) lists every downloaded file used for the test run
with its message count and SHA-256, to compare against.

## No leakage

The rule: nothing a forecast for time t uses, and nothing its model is trained on, comes from after t
([ADR-0001](docs/adr/0001-as-of-data-model.md)). What was done to hold to it, and the one assumption it rests on:

- **Features use only what was known before t.** An alert still on at t counts as ongoing, however long it later
  lasted. Tests compare the fast feature code with a naive "what was known at t" computation, and check that
  rewriting everything after t changes no feature at t.
- **Training labels must be known by the cutoff:** a model retrained at cutoff C only uses rows with t + h ≤ C,
  each with its label as known at C (a correction posted later is not applied yet).
- **Validation never sees the test period:** a forecast is scored only if its target hour lies inside its own period.
- **Choices are made on validation only,** each by a rule committed before the experiment ran; the test period was
  run with the configuration read from the validation decisions, and no choice was changed because of its scores
  (it was rerun twice after bug fixes, see Results).
- **Corrections are not known early.** An alert later retracted as a mistake counts as on alert until the
  retraction was posted; only the labels drop it ([ADR-0007](docs/adr/0007-published-vs-corrected-history.md)).
- **The unit list is fixed in advance,** so a raion's first alert never changes country or neighbour features before it.
- **Edited messages.** Telegram returns only a message's latest text, dated at its original posting time. In the
  label channel every message has been edited, mostly in two channel-wide re-edits (2024-01-30 and 2025-06-27)
  and, since July 2025, once more after posting: half within 29 seconds, but 13% more than 5 minutes later and
  2.7% more than an hour later. Sampled late edits keep the channel's fixed wording, so edits are treated as not
  changing an alert's kind or place. **This is an assumption, not a guarantee:** Telegram keeps no edit history, so
  what the channel said at t cannot be replayed. About 18,000 of 187,000 alert messages since November 2025 were
  edited after the next whole hour, so a forecast could in principle have used a later edit's wording. We found
  no case where an edit changed an alert's kind or place, and the month-by-month comparison with the independent
  official channel (below) is an approximate cross-check. **How much it could matter:** in a pessimistic replay where
  every message counts only from its last edit, validation Brier rises by at most 0.0036 on average (3%): 0.0087 at
  one hour ahead, 0.0008 for raions calm at t ([`scripts/replay_edit_times.py`](scripts/replay_edit_times.py),
  [`provenance_replay.csv`](reports/validation/provenance_replay.csv)). That is an upper bound: it also delays every
  alert start by the edit delay, although the alert was announced when posted.
  Telegram threat posts follow a stricter rule: a post counts only from its last edit, so a post edited at or
  after t is not used at t (ADR-0007).

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
- Three parser cases were found this way (or by review) and fixed: messages with several sections (since July 2026);
  "red alert over, yellow continues" with no alert open; all-clears that start with a time ("15:57 | Відбій тривоги").

Missing messages cannot be recovered, so the message-level check still fails. **At the level the model uses,
"was this raion on alert in this hour", the labels agree with the official record on 99.7–99.8% of known
raion-hours per year (2023–2026), and on at least 98.8% in every month** (the lowest is November 2025, the month
of the official channel's own broken alert). Unknown hours: 0.3% of raion-hours in 2023, 3.6% in 2026; 1.2% of the
validation period and 0.4% of the test period. After the official channel stopped (2026-09-07), the test period
is checked against alerts.in.ua instead: every month passes (98.4–99.96%), with 99.84% hour-level agreement
([`alerts_in_ua_test_period_hourly.csv`](reports/label_quality/alerts_in_ua_test_period_hourly.csv)).

## Limitations

- **Little skill a day ahead.** 12–24 hours ahead, the model is not measurably better than each raion's alert share
  over the last four weeks (Results). Its value is in the next few hours.
- **The edited-messages assumption** (No leakage): labels use each message's latest text at its original time.
- **The "h = 1" forecast is for the hour starting at t,** so it is not an alert-start forecast, and 30% of its rows are
  already on alert. A true alert-start target (a new alert beginning in the next hour) would be a different, harder
  task, and was not tried.
- **One forecast spec.** Only raion × hour × "any alert in the hour" is built. Other specs (oblasts, minutes on alert
  per day) are designed for but not implemented ([ADR-0002](docs/adr/0002-one-main-spec.md)).
- **City alert zones other than Kyiv** (Kharkiv, Zaporizhzhia, Nikopol, Marhanets, Kramatorsk, Kryvyi Rih) are folded
  into their raions. The check that would decide whether they deserve their own units (Q16) was not completed.
  Two quick counts put Kharkiv city's state apart from Kharkivskyi raion's in about 9% of test hours, above the 5%
  threshold we set, so it probably deserves its own unit.
- **Labels come from a volunteer channel.** They miss some messages (above). Hours that can't be known are left
  out, but some errors surely remain.
- **Scores leave out the hours with unknown labels,** which are the hours after a missed all-clear, when the history
  itself was wrong. Where the independent official record shows the outcome for such an hour (312 rows per
  horizon in the validation weeks it covers, 0.4% of the scored rows), the model's Brier score is 0.30 at one hour
  and 0.24–0.26 at 6–24 hours, against 0.07 and 0.12 on normal rows
  ([`unknown_hours_check.csv`](reports/validation/unknown_hours_check.csv)). The reported scores hold for hours the
  history describes correctly, not for source failures.
- **The test period is four weeks of one season,** shortly after alerts got yellow and red levels (September 2026;
  both count as an alert here). Results in other periods may differ.
- **Telegram extraction is rule-based, on two channels and at hourly resolution,** and was never checked against a
  hand-labelled sample. Known gap: a line that mixes an all-clear for one region with a threat for another can lose
  the threat. Other channels, a language model, or forecasts by the minute might find a signal that we didn't.
- **Ideas not tried:** blending the model with Recent rate by state at 12–24 hours (where they tie); Kharkiv city as
  its own unit; an alert-start target; a hand-checked sample of the Telegram extraction.

## Sources and attribution

- Alert labels: Telegram channel [`@UkraineAlarmSignal`](https://t.me/UkraineAlarmSignal) (єТривога), checked against the official channel [`@air_alert_ua`](https://t.me/air_alert_ua) and [alerts.in.ua](https://alerts.in.ua).
- Threat posts: Telegram channels [`@kpszsu`](https://t.me/kpszsu) (Air Force of the Armed Forces of Ukraine) and [`@deraketaua`](https://t.me/deraketaua).
- Raion list: [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) by Vadym Klymenko (MIT).
- Raion borders for the neighbour table: OCHA, [Ukraine – Subnational Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-ukr), HDX, CC BY-IGO. Only the derived neighbour pairs (`src/airalert/reference/neighbours.csv`) are in this repo.
