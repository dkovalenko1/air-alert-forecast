# Design: air raid alert forecasting

Status: **built and evaluated** (single test run on 2026-10-13; results in the README). Last updated 2026-10-13.

Related documents:
- Terms used here are defined in [CONTEXT.md](../CONTEXT.md).
- Hard-to-reverse decisions are recorded in [docs/adr/](adr/).

## 1. Product overview

**What it is.** A Python tool that forecasts air raid alerts in Ukraine. At a forecast time t it uses only information published before t: the alert history and Telegram threat posts. For each unit (raion, or Kyiv city) it outputs the probability of an alert in each of the coming hours.

**What question it answers.** "How likely is an alert here in the next hour, this evening, or this time tomorrow?" This helps people, schools and businesses plan. It is not a warning system and does not replace official alerts.

**What it can do:**

| Command | Result |
|---|---|
| `download-telegram` | Download new messages from the four channels. Incremental. |
| `build-labels`, `build-telegram` | Rebuild the alert periods and the threat events from the downloads. |
| `train [--cutoff …]` | Fit the configuration chosen on validation on everything known at a cutoff, and save it (about a minute). |
| `predict [--at …]` | Load the saved model and print probabilities per unit × horizon. Takes seconds and does no retraining. Refuses a time before the model's cutoff. |
| `backtest validation`, `experiment …`, `test-run`, `test-report` | Walk-forward evaluation against the baselines: validation, the experiments of section 3, and the single test run. |

Only the main spec (raion × 1 h × Active) is built; the code is organised by spec, but other specs are not implemented. Backtests print an estimate of the time left after the first fit.

Example of `predict` output (real, abridged):

```
Forecast time: 2026-10-13 00:00 Kyiv (data up to 2026-10-13 00:50); model: lightgbm, trained at 2026-10-13 00:00 Kyiv
                                       unit  on alert  h=1  h=2  h=3  h=6  h=12  h=24
       Харківська область/Харківський район      True 1.00 0.88 0.86 0.85  0.91  0.90
             Одеська область/Одеський район     False 0.42 0.36 0.41 0.42  0.47  0.52
         Львівська область/Львівський район     False 0.01 0.02 0.01 0.01  0.03  0.05
```

**Task constraints (from the KSE brief):**
- Use the most recent data, and evaluate on the latest period.
- No leakage: nothing used for training or as an input may come from after t.
- Optional text or media data, under the same rule.
- Submission is the GitHub repo plus a full export of the AI conversation. Secrets therefore never go into chat; they go only into the git-ignored `.env`.

## 2. How it works

```
sources ─► raw layer ─► canonical tables ─► spec builder ─► models ─► forecasts / backtest reports
           (as fetched,  (alert intervals,   (slots, labels,   (baselines, logistic regression,
           never edited)  threat posts; each  as-of features)   LightGBM, LightGBM + Telegram)
                          with an availability
                          time)
```

- **Raw layer:** responses are stored exactly as fetched, and everything else is rebuilt from them ([ADR-0001](adr/0001-as-of-data-model.md)).
- **As-of view:** features read data only through one function that returns what was known strictly before t. Alerts in progress at t are clipped at t ([ADR-0001](adr/0001-as-of-data-model.md)).
- **Forecast spec:** target × unit level × step × horizons. One main spec is analysed in depth ([ADR-0002](adr/0002-one-main-spec.md)).
- **Alert sources:** one alert definition per spec, from training through test ([ADR-0003](adr/0003-one-alert-source-per-spec.md)).
- **Unit levels are views of one alert table.** Each alert record has a location at one level: oblast ⊃ raion ⊃ hromada. Raion and oblast specs use the same records; only the mapping from records to units differs.
  - **Raion unit:** on alert when there is an alert for that raion, or an oblast-wide alert for its oblast. Before autumn 2025 most alerts were oblast-wide, so all raions of an oblast move together in that period.
  - **Oblast unit:** derived from the same records (definition: Q6).
  - **Kyiv city:** its own unit at both levels.
  - **Unit registry:** a single table lists every unit with its name in each source. The official data uses English transliteration, alerts.in.ua uses Ukrainian names and numeric UIDs, and some raions were renamed (e.g. Novomoskovskyi → Samarivskyi). Every source maps through this table.
- **Stages:** update data → build → train → predict. A backtest repeats train + predict over many training cutoffs. Training is the expensive stage; prediction loads saved models.

## 3. Evaluation protocol

- **Walk-forward backtest:** train on everything before a cutoff, forecast the next period, move the cutoff forward, repeat.
- **Labels must also be known by the cutoff.** A training row for forecast time t and horizon h has its label at t + h. So a model trained at cutoff C may only use rows with t + h ≤ C. Otherwise the last training rows carry outcomes from after the cutoff. Each label is also taken as known at C: a retraction posted after C does not correct it yet (ADR-0007).
- **Each period is scored on its own outcomes.** A forecast is scored only if its target hour lies inside its period, so validation scores never depend on outcomes in the test period.
- **Periods:** training | validation | test.
  - Model and setting choices are made on validation.
  - The test period is the latest period, and it is used once, for the final report.
- **Baselines:** Persistence and Historical rate. Every model must be compared against both.
- **Metrics, per horizon:**
  - Brier score (primary);
  - log loss, ROC-AUC, PR-AUC;
  - calibration plot.
- **Calm at t / On alert at t:** results are reported separately. The calm group measures how well onsets are predicted.
- **Ablation:** compare with and without Telegram features on the same period.
- **Reporting:** report every spec that was run, not only the best.

### Validation experiments (protocol fixed 2026-10-13, before any of them ran)

All three run on the validation period only (2026-08-17 → 2026-09-13), with the same weekly walk-forward as the final test. The test period is not loaded.

| Experiment | Variants | Models |
|---|---|---|
| **E1. Training start (Q9)** | 2023-01-01, **2025-08-01** (default), 2025-11-01 (Kyiv time) | logistic regression, LightGBM (default settings) |
| **E2. One shared model or one per unit (Q13)** | **shared** (default), one model per unit | logistic regression, LightGBM; training start from E1 |
| **E3. LightGBM settings** | **default** (300 trees, learning rate 0.05, 63 leaves, ≥100 rows per leaf); small trees (15 leaves); large trees (255 leaves, ≥200 rows per leaf); early stopping (up to 3,000 trees, number chosen on the last 14 days of the training rows); slow early stopping (learning rate 0.02, up to 5,000 trees); no unit identity (raion and oblast left out) | LightGBM; training start from E1, shared or per unit from E2 |

- **Selection rule:** the winner is the variant with the lowest Brier score averaged over the six horizons, on all validation rows and all units.
  - If it beats the default by less than 0.0005, the default is kept, because it is simpler.
  - Model choices are made per model: logistic regression and LightGBM each get their own winner in E1 and E2.
- **Reporting:** every variant's metrics are reported (`reports/validation/experiments/`), not only the winner's: Brier, log loss, ROC-AUC, PR-AUC per horizon, for all rows and for units calm at t.
- **Order:** E1 → E2 → E3, each using the earlier winners.

**E4. With or without Telegram (Q12)** (protocol fixed 2026-10-13, before it ran):

- **Variants:** each model as chosen by E1–E3, without (default) and with the Telegram threat-post features (`telegram_features.py`). Same grid, same weeks, same training rows; the only difference is the feature set.
- **Models:** logistic regression and LightGBM.
- **Selection rule:** the same as E1–E3. Telegram stays in a model only if it lowers that model's Brier score, averaged over the six horizons, by at least 0.0005.
- **Reported in addition:** per horizon and for units calm at t, where the gain is expected (short horizons, alert starts).

**E5. A small, targeted Telegram set (Q12)** (protocol fixed 2026-10-13, after E4's result and before E5 ran):

- **Why:** E4's 47 Telegram features did not pass the threshold (logistic −0.00032, LightGBM worse). E5 tests whether a few features, each chosen for a stated reason, do better. E5 was designed after seeing E4, and it is the second Telegram try on the same validation weeks, so a pass is weaker evidence than E1–E4. Both results are reported.
- **Features (7):**
  - posts naming the unit itself, any threat, last 60 min (a threat heading for this raion);
  - posts naming the unit's whole oblast, any threat, last 60 min;
  - posts naming a neighbouring unit, any threat, last 60 min (as in E4);
  - minutes since the last post naming the unit (as in E4);
  - nationwide drone posts, last 3 h (how big the current drone wave is);
  - nationwide missile and ballistic posts, last 3 h;
  - nationwide aviation posts, last 6 h (strategic bombers and MiG-31 often take off hours before a missile strike, which matters for the longer horizons).
- **Variants:** each model as chosen by E1–E3, without (default) and with these 7 features. Same grid, weeks and training rows.
- **Selection rule:** the same as E4. The targeted set stays in a model only if it lowers that model's mean Brier score by at least 0.0005.
- **No further Telegram variants** are tried on validation after E5, whatever it shows.
- **Test run:** each model is reported on the test period with and without the targeted set (the ablation above). The variant chosen here is the main result, so the test shows Telegram's effect whichever way E5 goes.

## 4. Decisions

Hard-to-reverse decisions (ADRs):

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-as-of-data-model.md) | Every record has an availability time; features read data only through the as-of view | accepted |
| [0002](adr/0002-one-main-spec.md) | Code accepts any forecast spec; one main spec is analysed in depth | accepted |
| [0003](adr/0003-one-alert-source-per-spec.md) | One alert definition per spec, from training through test | accepted |
| [0004](adr/0004-telegram-via-api.md) | Telegram posts are downloaded through the official API (Telethon) | accepted |
| [0005](adr/0005-rebuild-alert-history-from-channel.md) | The alert history is rebuilt from the `@air_alert_ua` messages, not taken from the published CSV | superseded by 0006 (parsing rules still apply) |
| [0006](adr/0006-labels-from-volunteer-channel.md) | Alert labels come from `@UkraineAlarmSignal` from 2023; the official channel is the validation reference | accepted |
| [0007](adr/0007-published-vs-corrected-history.md) | Features see the alert history as published; labels see it as corrected. Edited messages keep their posting time | accepted |

Decision log (smaller choices that are easy to reverse):

- **Model progression:** baselines → logistic regression → LightGBM → LightGBM + Telegram. LightGBM because the data is tabular, there are about 1M rows per horizon, it learns interactions on its own, it retrains fast, and SHAP explains it. *(2026-10-12)*
- **Adopted from Pavlyshenko & Pavlyshenko (2024, arXiv 2411.14625):**
  - neighbouring units' alert state and recent history;
  - duration of the current alert so far (0 if calm);
  - hour, weekday and month;
  - a comparison of one shared model against one model per unit.

  *(2026-10-12)*
- **Not adopted:** the "days since start" trend feature (`ndays`). Trees cannot extrapolate it. Drift is handled instead with the training window, recency weighting and frequent retraining. It may be tested as an ablation. *(2026-10-12)*
- **Rejected:** the Kaggle "Massive Missile Attacks on Ukraine" dataset. Its expected impact is small: it overlaps with the alert history, and its reports appear the morning after, so it cannot help at short horizons. It also needs its own leakage handling. *(2026-10-12)*
- **Rejected:** the UkraineAlarm API, because key requests are answered too slowly. *(2026-10-12)*
- **Complete raw downloads:** downloads cover every location, including ones we won't forecast. Exclusions happen when units are built, so they are visible and reversible. *(2026-10-12)*
- **Excluded from forecasting:** Luhansk oblast, Crimea and Sevastopol. Their alerts are permanent, so there is nothing to predict. *(2026-10-12)*
- **Time zones:** timestamps are stored in UTC. Calendar features (hour, weekday) are computed in Europe/Kyiv local time, so they follow daylight saving. *(2026-10-12)*
- **Validation experiments, results** (mean Brier over the six horizons; files in `reports/validation/experiments/`): *(2026-10-13)*
  - **E1, training start:** 2025-11-01 for both models (logistic 0.11013, LightGBM 0.11046), ahead of 2023-01-01 (0.11226 / 0.11231) and the 2025-08-01 default (0.11245 / 0.11268). Likely reason: the alert pattern changes over time, so older rows describe a regime that no longer applies.
  - **E2, shared or per unit:** shared for both (logistic 0.11013 vs 0.11359 per unit; LightGBM 0.11046 vs 0.12209). Likely reason: one raion's own rows (about 7,000 hours since November 2025) are too few to learn from alone.
  - **E3, LightGBM settings:** no unit identity (0.10881) beats the default (0.11046); the other settings fall between 0.11011 and 0.11174. Likely reason: without raion and oblast names the model must rely on what is happening around the unit, which carries over to new weeks better than a raion's own past.
  - **E4, all Telegram features:** left out of both models. Logistic gains 0.00032 (0.11013 → 0.10981), below the 0.0005 threshold; LightGBM gets worse (0.10881 → 0.10910).
  - **E5, targeted Telegram set (7 features):** left out of both models. Logistic gains 0.00037 (0.11013 → 0.10976), below the threshold; LightGBM gets worse (0.10881 → 0.10948), at every horizon. Telegram threat posts add little beyond the alert history of the unit and its neighbours at hourly resolution. Per E5's protocol, no further Telegram variants are tried on validation.
  - **Configuration for the test run:** training from 2025-11-01, one shared model, LightGBM without unit identity, alert-history features only; logistic regression and both baselines alongside; each model also with the targeted Telegram set (ablation).
- **Test run, results** (2026-09-14 → 2026-10-12, run once on 2026-10-13; `reports/test/`; mean Brier over the six horizons, 95% intervals from resampling days): *(2026-10-13)*
  - **LightGBM 0.1180**, logistic 0.1217, Historical rate 0.1576, Persistence 0.2148. LightGBM beats logistic by 0.0037 (0.0023 … 0.0054) and Historical rate by 0.0396 (0.0350 … 0.0440): 25% better than Historical rate (29% on validation).
  - **The validation ranking holds** in every week of the test period. Scores are higher than on validation (0.1088) partly because the test period had more alerts (42% of hours vs 38%).
  - **Telegram ablation:** the targeted set changes LightGBM by +0.0004 (−0.0002 … +0.0011) and logistic by −0.0002 (−0.0012 … +0.0006). Both intervals include zero, as on validation.
- **No separate notebook** (changes Q14): the README carries the narrative, with the figures drawn by `scripts/make_figures.py`, so the results live in one place. *(2026-10-13)*

## 5. Data sources (checked 2026-10-12)

| Source | Coverage | Location level | Use |
|---|---|---|---|
| Official ([Vadimkin dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), parsed from Telegram `@air_alert_ua`) | 2022-03-15 → 2026-09-07 | oblast / raion / hromada; 92% raion since Nov 2025 | Validation reference up to 2026-09-07 (ADR-0006) |
| [alerts.in.ua API](https://devs.alerts.in.ua/) history | Rolling one month, about 2026-09-12 → now | oblast / raion / hromada | Independent cross-check for the latest month |
| Volunteer (eTryvoga, same repo) | 2022-02-25 → now | oblast only | Not used for labels: its meaning drifted (see below) |
| Telegram `@UkraineAlarmSignal` (eTryvoga, 84k subscribers; the source behind the volunteer CSV) | 2022-02-25 → now, live | City/oblast until 2024; raion-level alerts from 2025; yellow/red levels from Sept 2026 | **Label source from 2023-01-01** (ADR-0006) |
| Telegram `@kpszsu` (Air Force), `@deraketaua` (volunteer monitoring) | 2022 → now | Text mentioning oblasts, raions and cities | Threat-post features |

Known facts that shape the design:
- **`@air_alert_ua` is silent:** the channel has posted nothing since 2026-09-07 06:09 UTC, and no reason has been published.
- **Raion gap:** no raion-level source in this table covers 2026-09-07 to about 2026-09-12 (the label channel `@UkraineAlarmSignal`, adopted later by ADR-0006, does).
- **Sources disagree:** official and volunteer data agree on only 80.6% of oblast-hours (June–September 2026). In Volyn, official data has 81 alerts and 41 h under alert; volunteer data has 355 alerts and 405 h.
- **Frontline oblasts:** if "oblast on alert" means "any raion on alert", frontline oblasts are on alert in about 100% of hours (Dnipropetrovsk 100%, Donetsk 99.8%, Kharkiv 99.4%; June–September 2026). These figures still include the non-air-raid hromada records described below.
- **Permanent alerts:** Luhansk (since 2022-04-04) and Crimea (since 2022-12-10) are not in the datasets. The official data has only 2 Luhansk records, both from 2022.
- **Near-permanent raions:** in June–September 2026, 16 of 118 raions were under a whole-raion alert in at least 90% of hours: all 8 Donetsk raions (99.2%), 5 Zaporizhzhia raions (98.3%) and others. Several of them are occupied.
- **The official CSV has exact duplicate rows:** 113,845 of 299,275 rows (38%), spread over every year (2026: 6,954). Shares of time under alert are not affected, because overlapping intervals merge. Earlier counts of rows were checked again without the duplicates: level mix since November 2025 is 92% raion / 7% hromada / 1% oblast, and the Volyn counts are unchanged.
- **Hromada-level records are not all air raid alerts** (about 7% of records since November 2025). The original channel messages show:
  - **Kharkiv and Zaporizhzhia cities:** their own air raid alert zones ("Повітряна тривога в м. …"). During an alert, extra 🔴🔴 escalation messages follow: "Загроза ударних БпЛА" (strike drones), "Загроза керованих авіабомб" (guided bombs), "Ракетна загроза" (missiles).
  - **Nikopol, Marhanets:** "Загроза артобстрілу" (artillery shelling threat), ended by "Відбій загрози артобстрілу". These are not air raid alerts.
  - **Pokrovsk, Vovchansk, Lypetska:** urban-fighting warnings. These are not air raid alerts.
- **The official dataset's parser** ([source](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/processors/official_channel_processor.py)) has two problems. The CSV keeps no message type, so neither can be undone from the CSV alone.
  1. **Every 🔴 message counts as an alert start**, and every 🟢 message as an end. The only message it ignores is "Тестовий Регіон". So artillery and urban-fighting warnings are recorded as air raid alerts.
     - Row-level check: the CSV rows for Nikopol on 2026-09-06, 16:12:07–16:44:22 and 18:57:14–03:59:44 UTC, match the "Загроза артобстрілу" → "Відбій загрози артобстрілу" messages to the second.
     - This is why Nikopol raion looked like it was on alert in 99.9% of hours.
  2. **A repeated 🔴 message for a location that is already on alert overwrites the alert's start time.** If the previous start was more than 3 h earlier, the parser instead closes the earlier alert with an imputed end at start + 1 h.
     - Example: the Zaporizhzhia city alert from 21:00 to 23:48 UTC on 2026-09-06 (2 h 48 min) is recorded as 23:44:35–23:48:49 (4 min), because escalation messages kept resetting its start.
     - Only 255 unique rows (0.14%) have the imputed 1 h duration. The amount of truncation can't be measured without the raw messages.
- **Scope of these checks:** they cover the latest ~20 messages per search on the channel's public web preview (early September 2026), plus row-level matches for Nikopol, Kharkiv and Zaporizhzhia. We know the mechanisms exist; we have not measured how many rows they affect across the full history.
- **alerts.in.ua types its alerts:** its `alert_type` field takes one of five values: `air_raid`, `artillery_shelling`, `urban_fights`, `chemical`, `nuclear` (per its Python client). There is no type for in-alert escalations or kamikaze-drone attacks.

### Yellow and red alert levels (September 2026)

- **What changed:** air threats now have two levels.
  - **Yellow:** "Дронова небезпека" (drone danger, e.g. 1–2 Shaheds). Businesses and transport may keep working.
  - **Red:** massed-drone, missile or missile-drone threat, including ballistic missiles. People must go to shelter.
- **No all-clear between levels:** a new message replaces the previous one, and "Відбій" (green) means no threat at all.
- **Dates:**
  - Announced 2026-09-02.
  - Cabinet resolution №1092 in force from 2026-09-04.
  - Kyiv switched on 2026-09-06 at 06:00 Kyiv time, in all districts except Pechersk, and only partly in Holosiivskyi and Svyatoshynskyi.
  - Where differentiation is technically impossible, the ordinary siren is still used.
  - The General Staff, Air Force and government reassign threats to levels weekly.
  - Sources: [Ukrinform](https://www.ukrinform.net/rubric-society/4160207-ukraine-to-introduce-yellow-and-red-air-threat-levels-zelensky.html), [LB.ua](https://lb.ua/society/2026/09/06/764092_kiievi_vidsogodni_diyatime.html), [Ukrainska Pravda](https://www.pravda.com.ua/news/2026/09/02/8051557/).
- **The channel never published levels.** Searches of `@air_alert_ua` find no "Дронова небезпека" or other level messages. Its message format has only one kind of alert.
- **Kyiv dropped out of the channel just before Kyiv switched.**
  - The last Kyiv message is "Відбій тривоги" at 2026-09-05 23:44 UTC, about 3 h before the switch.
  - Volunteer data still shows Kyiv alerts on 2026-09-06 and 2026-09-07.
  - The channel then went silent everywhere at 2026-09-07 06:09 UTC.
  - **Likely, not confirmed:** the switch to levels is why the channel stopped. No official statement has been found.
- **Alert volume didn't jump, but missing all-clears rose.**
  - Volunteer data shows no jump in weekly alert counts or alert-hours around the switch: about 900–1,180 alerts and 450–560 alert-hours per week, August–October 2026.
  - The share of volunteer alerts with no end message rose from about 5% before to 7–10% after the switch. This fits "no all-clear between levels".
- **alerts.in.ua describes levels:** its API docs give an `alert_level` (red or yellow) on active alerts. Whether its history endpoint carries the level too will be checked in the first snapshot.

### The volunteer CSV and its source channel `@UkraineAlarmSignal`

- **The CSV's format never changed:** its columns are `region, started_at, finished_at, naive`, as in 2022.
- **Its meaning changed several times.** Its parser ([source](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/processors/volunteer_etryvoga_processor.py)) was last changed in June 2024. It reads the channel by keyword:
  - **Region:** the first oblast keyword found anywhere in the message.
  - **Start:** any of "тривога", "загроза", "небезпека", "в укриття"…
  - **End:** "відбій"…
  - The output is oblast-level only.
- **Meanwhile, the channel's content changed:**
  - **2024:** city-level warnings start ("🛸 Загроза застосування БПЛА", "💣 Загроза застосування КАБів", "💥 Загроза обстрілу").
  - **2025:** raion-level alerts start.
  - **September 2026:** yellow/red levels and messages listing several raions.
- **What that does to the CSV:**
  - City drone, guided-bomb and shelling warnings are counted as oblast air raid alerts. This explains Volyn: official data has 81 alerts and 41 h under alert, volunteer data has 355 alerts and 405 h.
  - A message listing raions in several oblasts is credited to one oblast only.
  - One raion's all-clear ends the whole oblast's alert.
  - A yellow→red change counts as a new start, which overwrites the start time.
- **Conclusion:** the CSV's format is stable, but its meaning is not, so it is not used for labels.
- **The channel itself mirrors the official raion alerts.** Check window: 2026-09-06 12:00–18:00 UTC.
  - Every official raion-level start (88/88) and end (102/102) also appears in the channel, within 3 minutes. The median delay is 17 s for starts and 25 s for ends.
  - This holds after normalising apostrophes (ʼ/’/') and one renamed raion (Красноградський → Берестинський).
  - The channel has extra starts that aren't in the official data (32 of 124). In the examples checked, these are messages listing several raions and level changes.
  - The channel also carries its own city-level warnings (🛸 drones, 💣 guided bombs, 💥 shelling, ⚠️ "ЗМІ повідомляють про вибухи"). These are not air raid alerts.
  - It already posted "Червоний рівень тривоги" for Dnipropetrovsk raions on 2026-09-06. At the same time, `@air_alert_ua` still posted plain "Повітряна тривога".

### First parse of `@UkraineAlarmSignal` (Desktop export, 2023-01 → 2025-07-04)

- **The channel is run by hand.** Volunteers post every alert manually ("вручну відправляючи кожне сповіщення"). The channel also posts apologies for mistakes ("Просимо вибачення, сталась технічна помилка… Тривога в області відсутня").
- **Every alert message has a readable place.** 103,759 alert start/all-clear messages; for all of them the place could be read.
- **What else gets classified as unknown:** 1–1.8% of messages per year, mostly volunteer recruitment, fundraising and evacuation calls.
- **Edge cases, out of 50,727 alerts (median 43 min):**
  - 1,888 repeated starts while already on alert (1.9% of alerts);
  - 401 all-clears with no open alert;
  - 22 retracted alerts. Only apologies saying the air raid alert is absent count as retractions; most apologies retract the channel's own warnings ("Загроза БПЛА в місті відсутня") and change no alert. Another 6 air raid apologies arrive after an all-clear; those alerts are kept.
  - *Update 2026-10-13 (code review):* the full download has more wordings ("Приносимо вибачення…", the sentence on a third line, "тривоги в області не було"). Apologies are now read as a whole: 47 retractions so far, plus 31 apologies giving a late all-clear, which now end the alert when posted. A retracted alert counts as on alert until the retraction for features, and is dropped from labels ([ADR-0007](adr/0007-published-vs-corrected-history.md)).
  - 99 "тривога триває" messages with no open alert.
- **Long alerts.**
  - Oblast-level alerts over 12 h match the official record closely: Sumy 45 vs 45, Kharkiv 39 vs 36, Poltava 25 vs 21, Chernihiv 13 vs 12.
  - City zones are different: 529 city alerts last over 12 h, some for months (Zaporizhzhia from 2023-03 to 2025-04). These are missing all-clears, not real alerts.
- **City alert zones in the channel, 2023 → 2025-07:** Svitlovodsk 2,110 starts, Kyiv 1,073, Slavutych 869, Ochakiv 622, Kharkiv 393, Dobropillia 226, Zaporizhzhia 181, Kryvyi Rih 67, then a few more.

### Message types in `@air_alert_ua`

Sample: 2,000 messages from 100 web-preview pages spread evenly over the message IDs, 2022-03-14 → 2026-09-07. Shares are approximate. The proposed handling is part of Q15/Q17.

| Type | First line looks like | Share of sample | Since | Proposed handling |
|---|---|---|---|---|
| Air raid alert | 🔴 "Повітряна тривога в …" / 🟢 "Відбій тривоги в …" | ~82% | 2022 | **The target.** Each alert runs from start to all-clear. |
| Partial all-clear | 🟡 "Відбій тривоги в … область (тривога ще триває у якомусь районі)" | ~5% | 2022 | Closes the oblast-wide alert. Raion alerts that are still open stay open. |
| In-alert escalation | 🔴🔴 "Загроза ударних БпЛА" (strike drones) / "Загроза застосування КАБ" (guided bombs) / "Ракетна загроза" (missiles), ended by "Відбій загрози …" | ~2% | Oct 2024 | Not a new alert: it never changes the alert's start or end. Stored as an event. |
| Artillery shelling threat | 🔴 "Загроза артобстрілу" / 🟢🟡 "Відбій загрози артобстрілу" | ~7% | Oct 2022 | Not in the target. Stored. alerts.in.ua also has this type, so it could become a feature. |
| Kamikaze-drone attack (front-line FPV, incl. "типу Молнія", "на оптоволокні") | "Атака дронів-камікадзе" / "Відбій атаки дронів-камікадзе" | ~2% | 2025 | Not in the target. Stored. |
| Urban fighting | "… вуличних боїв …" | <0.5% (not in sample; found by search) | — | Not in the target. Stored. |
| Radiation / chemical | "Радіаційна загроза …" | ~0.1% | 2025 | Not in the target. Stored. |
| Service / info | "УВАГА!!!", "БЕЗКОШТОВНА ЕВАКУАЦІЯ", "Channel created" | ~2% | — | Ignored, but counted. |

- **Formats change over time.** For example, 2025 escalations have no 🔴 in the first line, while 2026 ones start with 🔴🔴. Some service messages start with 🟢 or 🟡, so the old "emoji = start/end" rule mis-reads them. Our parser classifies by wording, not emoji.
- **Unknown messages:** any message the parser can't recognise is logged and counted, and is never treated as an alert.
- **City zone vs raion:** Kharkiv city's own alerts mostly fall inside Kharkivskyi raion alerts. The city is on alert while the raion is not in 0.5% of the time; the raion is on alert while the city is not in 11.7% (June–September 2026).
- **Telegram volume grows:** posting on `@kpszsu` grew a lot over time, from about 2k posts by October 2022 to about 86k by October 2026, so the meaning of Telegram features changes over time.

## 6. Open questions (to answer before building)

Question numbers are stable IDs, not an order. When a question is answered, it is marked *Decided* and its decision is recorded in the decision log or in an ADR.

### Data

- **Q1. Does an oblast-level alerts.in.ua history request return raion-level alerts?** *Answered 2026-10-12: yes.*
  - A request for Dnipropetrovsk oblast returned 2,304 alerts at raion, city and hromada level, so 27 requests cover the country.
  - Each alert has `alert_type` (`air_raid`, `artillery_shelling`) and `alert_level` (`red`, `yellow`).
  - The history reaches back to about 2026-09-11.
  - Caveats: alerts from August are also `red`, so `red` may simply be the default. City and hromada alerts duplicate each other (e.g. "м. Кривий Ріг" and "Криворізька територіальна громада", 204 each).
  - Why it matters: if it doesn't, we need about 125 raion requests (about 1 h) instead of 27 (about 14 min).
  - How we'll decide: inspect the first snapshot.
- **Q2. Is alerts.in.ua consistent with the label source?** *Decided 2026-10-12: used as the cross-check for the test period.* (ADR-0006)
  - **Why it is still needed:** the official channel is silent after 2026-09-07. For the whole test period, alerts.in.ua is the only independent check of our labels. It is also the fallback if the channel stops before submission.
  - **First check:** on 2026-10-10, 12:00–18:00 UTC, for Dnipropetrovsk raions:
    - 23/23 alerts.in.ua starts and 25/26 ends were found in the channel, median delay 16–18 s;
    - 24/25 channel starts were found in alerts.in.ua.
  - **Full check:** repeat over the whole test period with the 27-request snapshot.
- **Q3. Fallback if Q2 fails.** *No longer needed: the label source is `@UkraineAlarmSignal` (ADR-0006).*
  - Options: an oblast-level main spec built from one consistent source, or a shorter, alerts.in.ua-only raion period.
- **Q19. Which source do the labels come from?** *Decided 2026-10-12: `@UkraineAlarmSignal` from 2023-01-01* ([ADR-0006](adr/0006-labels-from-volunteer-channel.md)). (See "The volunteer CSV and its source channel" in section 5.)
  - **Channel history:**
    - Created 2022-02-25 at 16:11 UTC; its first alert was "Тривога в Києві" at 16:36. That is 18 days before the official record starts.
    - Formats:
      - 2022–2024: oblast-level messages ("Харківська обл. | Тривога - Усі в укриття"), plus some city names in 2022.
      - 2025: raion-level messages phase in (about 14% of alert messages in H1 2025, 79% in H2, about 96% in 2026).
      - September 2026: yellow/red levels.
  - **Agreement with the official record.** Spot checks after removing duplicates, with a ±5 min match window:

    | Window (UTC) | Level | Official starts found in channel | Official ends found in channel | Median delay |
    |---|---|---|---|---|
    | 2022-06-15 00–12 | oblast | 5/7 | 6/9 | 9–27 s |
    | 2023-06-15 00–12 | oblast | 17/17 | 24/24 | 22–28 s |
    | 2024-06-15 00–12 | oblast | 35/35 | 47/47 | 33–41 s |
    | 2025-06-15 00–12 | oblast + raion | 23/24 | 23/24 | 5–7 s |
    | 2026-09-06 12–18 | raion | 88/88 | 102/102 | 17–25 s |

    In the other direction, almost every channel alert also appears in the official record. 2022 is weaker: some official alerts are missing, and some messages name cities instead of oblasts.
  - **Proposal: `@UkraineAlarmSignal` is the label source for the whole history from 2023-01-01.**
    - One source and one parser cover training and test, the strongest form of ADR-0003.
    - The official channel (ADR-0005) becomes the validation reference rather than the label source.
    - 2022 is left out: it is the weakest period for this channel and an early-war regime anyway. The main spec is raion-level, so it starts in 2025 regardless.
  - **Parsing:** our own type-aware rules, as in ADR-0005.
    - Air raid alerts only: "Повітряна тривога", the yellow/red levels, and "Відбій тривоги".
    - Its 🛸/💣/💥/⚠️ warnings are dropped from the labels.
  - **Why:**
    - It is live and raion-level, and it covers the 2026-09-07 → 09-12 gap.
    - It carries the levels (Q18).
    - It gives the overlap with the official record that alerts.in.ua cannot (ADR-0003).
  - **Before accepting it:** repeat the spot checks over whole months, e.g. one month per year plus August 2026 → 2026-09-07, using a Telethon download of both channels.
  - **alerts.in.ua:** becomes an independent cross-check for the latest period, which turns Q2 into a three-way check.
- **Q20. Pass mark for the label check, set before looking at the results.** *Decided 2026-10-12.*
  - **Also calibrated here: detecting a missing all-clear** (Q17). For each level, compare long channel alerts with the official alert for the same place, then choose the rule. City zones are the main case.
  - **What it's for:** ADR-0006 rests on 54 hours of spot checks. Before building on the channel, we compare it with the official channel month by month, from 2023-01 to 2026-08.
  - **Why it's fixed in advance:** with a pass mark chosen after seeing the results, any number can be explained away ("97% is fine").
  - **If a month fails:** we look for the cause (parser bug, channel outage, renamed places) and fix the parser, mark the period missing, or start training later.
  - **Output:** the per-month table goes into the README as evidence of label quality.
  - **Result (2026-10-13): fails.** After the fixes below, 33 of 176 month × kind × direction rows are below 98% (lowest 96.2%; 18 months, 15 rows in 2023). *Decided 2026-10-13: accept the labels, report the failure and its causes in the README (option 1), and also handle the channel's missed starts (option 2).* Causes found:
    - the official channel's outage, 2026-08-29 07:55 to 2026-09-02 16:48 Kyiv time: now left out of the comparison (any silence over 24 h; normal quiet spells last at most about 12 h);
    - a parser bug: messages with several sections (since 2026-07) lost all-clears; fixed;
    - the channel misses some all-clears: handled by the missing-all-clear rule (Q17), which makes those stretches unknown;
    - the channel misses some starts (it posts an all-clear for an alert it never announced): handled since 2026-10-13. An all-clear with no open alert, more than 20 min after the place's previous all-clear (closer ones are repeated posts), makes the 5 h before it (or since the place's previous message) unknown; in the official channel such all-clears followed a real alert in 90–98% of cases, and 95% of those alerts had started within 5 h. "Calm in our labels, on alert officially" fell to 0.04–0.29% of known raion-hours per month from 2025-08 (2.3% in 2025-11);
    - the official record has broken alerts of its own (e.g. 135 h in Lviv and Chernivtsi raions, 2025-11-19 to 11-24).
    - At the hour level, which is what the model uses, labels and the official record agree on 99.5–99.7% of known raion-hours per year (2023–2026) and at least 97.3% in every month (2025-11). `check-labels` writes this as `official_hourly_by_month.csv`.
  - **Pass mark:** in every month, at least 98% of official starts and ends are found in the channel within 5 min, and vice versa.
  - **Cost:**
    - Messages to download since 2023: about 303k for `@air_alert_ua` (to 2026-09-07) and about 607k for `@UkraineAlarmSignal`, which is needed for labels anyway.
    - Telethon waits about 1 s per 100 messages by default, so roughly 1 h and 2 h of unattended download.
    - The check itself reuses the label parsers; the comparison script and the run take about an hour.
    - Investigating failed months can't be estimated in advance.
- **Q4. Keeping the latest period current.** *Decided 2026-10-12: pinned dates.*
  - `update-data` downloads only messages newer than the last one we have. It runs once, shortly before the final evaluation.
  - The test period ends on the last full Sunday before that run.
  - The period dates are pinned in the configuration and recorded in the README, so a rerun reproduces the report. Graders can move the dates forward.
- **Q15. How do we keep only real air raid alerts in the labels, with correct start times?** (See "Hromada-level records" and "The official dataset's parser" in section 5.)
  - **(a) Re-parse `@air_alert_ua` ourselves** through Telethon (about 338k messages, roughly 1 h to download).
    - Keep each message's type.
    - Treat 🔴🔴 escalation messages as escalations, not as new starts.
    - Vadimkin's MIT-licensed parser is the starting point.
    - This also measures how many CSV rows the two parser problems affected.
  - **(b) Keep the official CSV, using raion- and oblast-level records only**, plus the hromada zones checked by hand to be air raid zones (Kharkiv and Zaporizhzhia cities).
    - Cheaper.
    - It relies on a sample showing that raion-level messages are air raid alerts only.
    - It keeps the overwritten start times wherever escalation messages occurred.
  - Either way, alerts.in.ua is filtered to `alert_type = air_raid`, so training and test share one definition (ADR-0003).
  - *Decided 2026-10-12: (a)* ([ADR-0005](adr/0005-rebuild-alert-history-from-channel.md)).

### Target

- **Q5. Main spec.** *Decided 2026-10-12* ([ADR-0002](adr/0002-one-main-spec.md)):
  - Main: raion × 1 h × Active, horizons 1–24, provided Q2 passes.
  - Secondary: raion × 1 day × Minutes (day-ahead planning). The oblast × 1 h × Active spec was dropped on 2026-10-12 (Q6).
- **Q6. Definition of "oblast on alert" since alerts became raion-level.** *Decided 2026-10-12: not needed. The oblast spec is dropped and we forecast raions only (plus Kyiv and any city units from Q16).*
  - Options: any raion; at least half of the raions; the raion containing the oblast centre; only alerts declared for the whole oblast.
  - **Only needed for the secondary oblast spec,** which is now dropped.
    - The question the oblast spec answered ("does a longer history help?") is already answered by the Q9 training-start experiment on the raion spec.
    - An oblast view, if wanted for display, is an aggregate of the raion forecasts, not a separate model.
- **Q7. Label rules.** *Decided 2026-10-12: any overlap makes a slot Active, and alerts are kept exactly as published.*
  - **Active:** a slot is Active if any alert overlaps it (decided 2026-10-12). Requiring at least 1 minute would barely matter: only 0.2% of raion alerts since 2025-08 lasted under 1 minute (median 53 min).
  - **No merging:** alerts are kept exactly as published (decided 2026-10-12). Merging changes nothing for Active or Minutes; it would only affect Onset and Time to alert.
    - Example: Synelnykivskyi raion had an all-clear at 13:18:10 and a new red alert at 13:19:40 on 2026-09-06.
    - If an Onset spec is run, we report how many alerts restart within 1/2/5/10 min of an all-clear, then decide.
  - **Naive volunteer alerts:** no longer relevant, because the volunteer CSV is not used (ADR-0006).
- **Q8. Units to report separately.** Permanent alerts are already excluded (see the decision log).
  - Near-permanent and occupied raions: 16 of 118 raions were under alert in at least 90% of hours in June–September 2026.
  - *Decided 2026-10-12:* keep them in training, because their state is useful neighbour information. Report metrics both with and without them, because they make every model look good.
  - Which raions count as near-permanent is decided on the validation period, not the test period. "On alert in an hour" means any alert during that hour, as the Active label counts it.
- **Q16. Should city zones with their own air raid alerts be units?** *Decided 2026-10-12: by the measured rule below.*
  - Kharkiv and Zaporizhzhia cities have their own air raid alert zones. Kharkiv city alerts mostly fall inside Kharkivskyi raion alerts, but differ from them in about 12% of the time.
  - Options: the city becomes its own unit (like Kyiv city), or it is folded into its raion.
  - **Candidates seen in the channel's city-level air raid messages:** Kharkiv, Zaporizhzhia, Nikopol, Marhanets, Kramatorsk. alerts.in.ua also lists Kryvyi Rih.
  - **Measured per candidate city:**
    1. **Label accuracy:** its alerts in the channel, compared with the official channel's records for that city up to 2026-09-07 and with alerts.in.ua `city` records after that. Same method as the Q20 check.
    2. **Added value:** the share of hours in which the city's state differs from its raion's. Kharkiv: about 12%. Zaporizhzhia: the raion is on alert while the city is not in 51% of hours (June–September 2026).
    3. **Enough data:** the number of city alerts per month in training.
  - **"Folded into its raion" means** the city's own alerts are ignored, and the raion's alerts stand for it. That way broken city records (missing all-clears) can't leak into raion labels.
  - **Rule, set before measuring:** a city becomes a unit if it passes Q20's pass mark, differs from its raion in at least 5% of hours, and has at least 20 alerts per month. Otherwise it is folded into its raion.
  - Forecast quality for city units is then reported separately in the backtest.
- **Q18. What counts as an air raid alert under the yellow/red system?** *Decided 2026-10-12: an alert of either level.* (See "Yellow and red alert levels" in section 5.)
  - **The problem:** every training alert comes from the old single-level system. The whole test period, about 2026-09-14 → 2026-10-11, falls under the new one. The definition changes exactly at the train/test boundary.
  - **Target = an alert of either level (yellow or red).** The old "Повітряна тривога" covered drones and missiles alike, so "either level" is the closest match to the old definition. There are only about 5 weeks of level data, and nearly all of it falls in the test period.
  - **Parsing:** a yellow↔red change continues the same alert (the Q17 repeated-start rule). An alert ends only at "Відбій тривоги".
  - **The level is stored as an attribute.** It is used for exploratory analysis and to split test results after the fact ("does the model catch red alerts as well as yellow ones?"). Using it only to group outcomes is not an input to the model, so there is no leakage.
  - **Red-only as the target:** not possible. There is no red-level history to train on.
  - **Level as a feature:** not now, because it doesn't exist before September 2026 (same rule as escalations in Q17).
  - **Checks:**
    - Does alerts.in.ua history carry `alert_level`?
    - Where and when did each region switch? Rollout is regional, and Kyiv switched on 2026-09-06.
    - Do alert counts and durations per raion jump at the switch? Extends Q2.
  - **Reporting:** the switch is reported as a limitation. Test results may also be shown per week, to see whether weekly retraining adapts to the new system.
- **Q17. Parsing rules.** *Decided 2026-10-12: all three proposals accepted* ([ADR-0005](adr/0005-rebuild-alert-history-from-channel.md)). (See "Message types" in section 5.)
  - **Repeated start for a location that is already on alert:** keep the original start. This is exactly what fixes the overwrite problem.
  - **Missing all-clear** (an alert never closed, then a new start hours later): the true end is unknown. Close the alert at its last known message and mark the gap as unknown. Unknown periods are not trained on or scored. No end time is invented (the old parser used start + 1 h).
    - *Implemented 2026-10-13, calibrated against the official channel (Q20) on data before validation:* a stretch between two messages about an open alert is unknown if a new start without a level arrives more than 20 min after the previous message (the official channel was calm in between in 86–100% of such cases), or if it is longer than the oblast's limit: twice the official 99th percentile of alert length, at least 12 h (`scripts/calibrate_missing_all_clear.py`, `reference/max_silence_hours.csv`). Features keep the alert as published; labels in an unknown stretch are missing unless a confirmed alert touches the hour; training at a cutoff sees the published label until the stretch could have been detected (ADR-0007).
    - *Effect, 2025-08 to 2026-08:* hour-level agreement with the official channel on known hours rose from 95.0–99.7% to 99.5–99.9% per month (97.2% in 2025-11, where the official record itself has a broken 135-hour alert). Unknown share: 0.2–7.9% of raion-hours per month; 1.0% in validation, 0.4% in test.
  - **Escalation with no open alert:** e.g. "Загроза керованих авіабомб" arrives when no "Повітряна тривога" is open for that location. Open an implied air raid alert, flag it, and count how often this happens.
  - **Non-target types as features:** only types that also exist in the test period can be features. After 2026-09-07 the only alert source is alerts.in.ua. It has artillery shelling and urban fighting, but no escalations or kamikaze-drone attacks. So escalations can be used for exploratory analysis and the write-up, but not as model inputs. Telegram threat posts (`@kpszsu`, `@deraketaua`) cover that information in both periods.

### Evaluation

- **Q9. Periods.** *Decided 2026-10-12:*

  | Period | Dates (Kyiv time) | Role |
  |---|---|---|
  | Training | from 2025-08-01 | The model learns from inputs and outcomes. 2025-08 is when raion alerts became the majority (69%, then 83% from September). |
  | Validation | Mon 2026-08-17 → Sun 2026-09-13 | All model and setting choices are made here. It includes the switch to yellow/red levels (2026-09-04 to 09-06). |
  | Test | Mon 2026-09-14 → Sun 2026-10-11 | Four full weeks, used once for the reported score. Entirely under the yellow/red system. |

  - **Weekly retraining inside validation and test:** each week is forecast by a model trained on everything before that week starts. Forecasts are made every hour, for every horizon.
  - **Training-start experiment on validation:** 2025-08-01 vs 2025-11-01 (fully raion-level) vs 2023-01-01 (oblast-wide alerts spread to all raions). The winner is used for the test period. Protocol: experiment E1 (section 3).
  - **Label cutoff:** a training row is only used if t + h ≤ the training cutoff (section 3).
- **Q10. Which horizons to train.** *Decided 2026-10-12: six horizons, 1, 2, 3, 6, 12 and 24 h.* All 24 would take about 1.5–5 h per backtest for the raion spec; six take about 25–75 min.

### Features

- **Q11. Neighbour map.** *Decided 2026-10-12: shared border plus same oblast.*
  - Raion boundaries come from OCHA's admin-2 maps on HDX; licence still to be checked. Neighbours are computed once and only the resulting table is committed.
  - Features use the share of neighbours on alert, not the count, because raions have different numbers of neighbours.
  - City units inherit their raion's neighbours.
  - Learned neighbours (raions whose alerts tend to come first, computed from training data only) are a later experiment, if there's time.
- **Q12. Telegram features.** *Decided 2026-10-12, as proposed:*
  - Channels: `@kpszsu` and `@deraketaua`.
  - Text is matched to places with a list of names in every grammatical case. Cities are mapped to raions through the official КАТОТТГ register.
  - No LLM extraction.
  - Features: threat posts for each unit and its neighbours over the last 15/60/180 min, split by type; time since the last mention; country-wide signals.
  - A post edited after t is not used for forecasts made before the edit.
  - The start date comes from a chart of posts per month.

  The original sub-questions:
  - Final channel list.
  - Matching text to units: a list of oblast, raion and city names in all grammatical cases, mapped to raions.
  - Feature types: threat posts per unit in the last 1/3/6 h, threat type (Shahed, jet drone, ballistic, cruise missile, MiG-31K, bombers).
  - Edited posts: if a post was edited at or after t, use the original or drop it?
  - From which date the Telegram model is trained, based on a plot of posts per month.
  - Exclude LLM-based extraction? A model trained on data from the test period could leak.
- **Q13. One shared model or one per unit:** decided by an experiment on the validation period. Default: shared. Protocol: experiment E2 (section 3).

### Product

- **Q14. Deliverable form.** *Decided 2026-10-12:*
  - CLI (`python -m airalert update-data | build | train | backtest | predict --spec …`).
  - Code in `src/airalert/`; `tests/` includes the leakage test from ADR-0001.
  - README with results per horizon against the baselines, the Q20 label-quality table, and limitations.
  - Charts generated by a script into `docs/figures/` as PNG, plus one narrative notebook.
  - Small derived tables (unit registry, neighbours) are committed; raw data and models are not.
  - `uv` + `pyproject.toml`, Python 3.12.
  - The AI conversation export is added to the repo as well (`docs/ai-conversation.md`). Before it is committed, it is checked for tokens and credentials.

## 7. Implementation plan

Status values: **done**, **running**, **open** (can start now), or **blocked** (named downloads still missing). "Develop now" means the code can be written against samples, the Desktop export or synthetic data. The real run waits for the full download.

| # | Step | Needs | Status |
|---|---|---|---|
| 1 | Download alerts.in.ua history | — | done (27 locations, 16,812 alerts) |
| 2 | Download Telegram channels (`@UkraineAlarmSignal`, `@air_alert_ua`, `@kpszsu`, `@deraketaua`) | login | done (to 2026-10-12: 652,627 / 338,361 / 85,427 / 60,451 messages) |
| 3 | **Unit registry:** every raion with its oblast and name variants (renames, apostrophes, Ukrainian and English) | official CSV, alerts.in.ua | done (`units.py`, 122 raions + Kyiv, renamed aliases) |
| 4 | **Label parser for `@UkraineAlarmSignal`:** message classifier per format period, and the alert state machine with the Q17 rules, producing the alert interval table | step 3; samples and Desktop export for development | done (`alarm_signal.py`, `alerts.py`, `labels.py`), incl. multi-section messages and the calibrated missing-all-clear rule |
| 5 | **Parser for `@air_alert_ua`:** the same rules, for validation | step 3; 2,000-message sample | code done (`official_channel.py`); full run after the download |
| 6 | **Q20 label check:** channel vs official, month by month, against the pass mark | steps 4–5 with full downloads | done: **fails** (33 of 176 rows below 98%); causes handled and reported in the README (decided 2026-10-13) |
| 7 | **Q2 cross-check:** channel vs alerts.in.ua over the test period | step 4 with full download | done: all rows pass (98.4–99.96%); 99.8% hour-level agreement |
| 8 | **Q16 city units:** apply the rule (label accuracy, difference from its raion, alerts per month) | steps 6–7 | not done before the test run: city zones other than Kyiv are folded into their raions (the Q16 fallback); left as future work |
| 9 | **Neighbour map (Q11):** raion borders from OCHA/HDX, plus a same-oblast table | step 3 | done (`reference/neighbours.csv`, 612 pairs) |
| 10 | **As-of view and the leakage test (ADR-0001):** features read only data known before t | — (synthetic data) | done (`features.py`, leakage tests) |
| 11 | **Grid, labels and alert-history features** for any forecast spec, including the label cutoff (t + h ≤ cutoff) | steps 4, 9, 10 | main spec done (`features.py`: raion × 1 h × Active, h = 1–24); other specs (e.g. the daily Minutes spec) planned; real run after the download |
| 12 | **Backtest harness:** weekly walk-forward, Persistence and Historical-rate baselines, metrics per horizon, calm/on-alert split, time estimate | step 11 | code done (`backtest.py`, `models.py`); results blocked |
| 13 | **Models:** logistic regression, LightGBM; training-start experiment (Q9) and shared vs per-unit (Q13) on validation | step 12 with real labels | done: E1–E3 (section 3); chosen: training from 2025-11-01, one shared model, LightGBM without unit identity |
| 14 | **Telegram features (Q12):** place-name list (КАТОТТГ register), threat-type rules, edit rule | place names: open; rules: `@kpszsu`/`@deraketaua` downloads (last in queue) | done (`places.py`, `threats.py`, `telegram_features.py`): 99% of `@kpszsu` and 93% of `@deraketaua` posts typed |
| 15 | **Telegram ablation** on validation, then the single test run | steps 13–14 | done: E4, E5 (Telegram not kept); test run 2026-10-13 (`reports/test/`) |
| 16 | **Final run:** pinned test period, figures, README, AI conversation export | everything | done 2026-10-13 (`train`/`predict` added; no separate notebook, see the decision log) |
