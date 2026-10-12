# Design: air raid alert forecasting

Status: **pre-build**. Last updated 2026-10-12.

Related documents:
- Terms used here are defined in [CONTEXT.md](../CONTEXT.md).
- Hard-to-reverse decisions are recorded in [docs/adr/](adr/).

## 1. Product overview

**What it is.** A Python tool that forecasts air raid alerts in Ukraine. At a forecast time t it uses only information published before t: the alert history and Telegram threat posts. For each unit (raion or oblast) it outputs the probability of an alert in each of the coming hours.

**What question it answers.** "How likely is an alert here in the next hour, this evening, or this time tomorrow?" This helps people, schools and businesses plan. It is not a warning system and does not replace official alerts.

**What it can do:**

| Command | Result |
|---|---|
| `update-data` | Download new alert history and Telegram posts. Incremental. |
| `train --spec …` | Fit the models for a forecast spec on everything before a training cutoff, and save them. |
| `predict --spec … --at <time>` | Load saved models and print probabilities per unit × horizon. Takes seconds and does no retraining. |
| `backtest --spec …` | Walk-forward evaluation on the latest period against baselines. Writes metric tables and plots. |

Long training and backtest runs print an estimated duration first. The estimate comes from a timed trial fit scaled to the full data size.

Example of `predict` output (the numbers are illustrative):

```
Forecast spec: raion × 1 h × Active   Forecast time: 2026-10-12 18:00 Kyiv (data up to 17:59)
Raion                 h=1    h=3    h=6    h=12   h=24
Kharkivskyi           0.83   0.66   0.58   0.49   0.55
Kyiv city             0.11   0.19   0.27   0.34   0.21
…
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

## 4. Decisions

Hard-to-reverse decisions (ADRs):

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-as-of-data-model.md) | Every record has an availability time; features read data only through the as-of view | accepted |
| [0002](adr/0002-one-main-spec.md) | Code accepts any forecast spec; one main spec is analysed in depth | accepted |
| [0003](adr/0003-one-alert-source-per-spec.md) | One alert definition per spec, from training through test | accepted |
| [0004](adr/0004-telegram-via-api.md) | Telegram posts are downloaded through the official API (Telethon) | accepted |
| [0005](adr/0005-rebuild-alert-history-from-channel.md) | The alert history is rebuilt from the `@air_alert_ua` messages, not taken from the published CSV | accepted |

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

## 5. Data sources (checked 2026-10-12)

| Source | Coverage | Location level | Use |
|---|---|---|---|
| Official ([Vadimkin dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), parsed from Telegram `@air_alert_ua`) | 2022-03-15 → 2026-09-07 | oblast / raion / hromada; about 90% raion since Nov 2025 | Main alert history |
| [alerts.in.ua API](https://devs.alerts.in.ua/) history | Rolling one month, about 2026-09-12 → now | oblast / raion / hromada | Fill the gap and cover the latest period. Snapshot it before the window moves on. |
| Volunteer (eTryvoga, same repo) | 2022-02-25 → now | oblast only | Cross-check reference |
| Telegram `@kpszsu` (Air Force), `@deraketaua` (volunteer monitoring) | 2022 → now | Text mentioning oblasts, raions and cities | Threat-post features |

Known facts that shape the design:
- **`@air_alert_ua` is silent:** the channel has posted nothing since 2026-09-07 06:09 UTC, and no reason has been published.
- **Raion gap:** no raion-level source covers 2026-09-07 to about 2026-09-12.
- **Sources disagree:** official and volunteer data agree on only 80.6% of oblast-hours (June–September 2026). In Volyn, official data has 81 alerts and 41 h under alert; volunteer data has 355 alerts and 405 h.
- **Frontline oblasts:** if "oblast on alert" means "any raion on alert", frontline oblasts are on alert in about 100% of hours (Dnipropetrovsk 100%, Donetsk 99.8%, Kharkiv 99.4%; June–September 2026). These figures still include the non-air-raid hromada records described below.
- **Permanent alerts:** Luhansk (since 2022-04-04) and Crimea (since 2022-12-10) are not in the datasets. The official data has only 2 Luhansk records, both from 2022.
- **Near-permanent raions:** in June–September 2026, 16 of 118 raions were under a whole-raion alert in at least 90% of hours: all 8 Donetsk raions (99.2%), 5 Zaporizhzhia raions (98.3%) and others. Several of them are occupied.
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
     - Only 0.12% of all rows have the imputed 1 h duration. The amount of truncation can't be measured without the raw messages.
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

- **Q1. Does an oblast-level alerts.in.ua history request return raion-level alerts?**
  - Why it matters: if it doesn't, we need about 125 raion requests (about 1 h) instead of 27 (about 14 min).
  - How we'll decide: inspect the first snapshot.
- **Q2. Is alerts.in.ua consistent with the official record?** No overlap period exists, so we check indirectly:
  - **(a)** compare agreement(alerts.in.ua, volunteer) after 2026-09-12 with agreement(official, volunteer) before 2026-09-07, per oblast;
  - **(b)** compare distributions across the boundary: alerts per raion per day, durations, and the mix of location levels.
  - If they match, the raion main spec goes ahead. If not, see Q3.
- **Q3. Fallback if Q2 fails.**
  - Options: an oblast-level main spec built from one consistent source, or a shorter, alerts.in.ua-only raion period.
- **Q4. Keeping the latest period current.**
  - Re-run the fetch before the final evaluation? Schedule regular alerts.in.ua snapshots so that history is not lost when the one-month window moves on?
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
  - Secondary: oblast × 1 h × Active (how much does the raion level gain over a longer history?) and raion × 1 day × Minutes (day-ahead planning).
- **Q6. Definition of "oblast on alert" since alerts became raion-level.**
  - Options: any raion; at least half of the raions; the raion containing the oblast centre; only alerts declared for the whole oblast.
- **Q7. Label rules.**
  - How much overlap makes a slot Active: any overlap, or at least 1 minute?
  - Are alerts separated by a short gap merged when counting onsets? If so, how short is "short"?
  - What to do with volunteer "naive" alerts, whose end time is imputed as start + 30 min?
- **Q8. Units to report separately.** Permanent alerts are already excluded (see the decision log).
  - Near-permanent and occupied raions: 16 of 118 raions were under alert in at least 90% of hours in June–September 2026.
  - Proposal: keep them in training, because their state is useful neighbour information. Report metrics both with and without them, because they make every model look good.
- **Q16. Should city zones with their own air raid alerts be units?**
  - Kharkiv and Zaporizhzhia cities have their own air raid alert zones. Kharkiv city alerts mostly fall inside Kharkivskyi raion alerts, but differ from them in about 12% of the time.
  - Options: the city becomes its own unit (like Kyiv city), or it is folded into its raion.
  - Depends on Q15, which determines which hromada zones carry real air raid alerts.
- **Q18. What counts as an air raid alert under the yellow/red system?** (See "Yellow and red alert levels" in section 5.)
  - **The problem:** every training alert comes from the old single-level system. The whole test period, about 2026-09-14 → 2026-10-11, falls under the new one. The definition changes exactly at the train/test boundary.
  - **Proposal: target = an alert of either level (yellow or red).** The old "Повітряна тривога" covered drones and missiles alike, so "either level" is the closest match to the old definition. Store the level as an attribute.
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
  - **Escalation with no open alert:** e.g. "Загроза керованих авіабомб" arrives when no "Повітряна тривога" is open for that location. Open an implied air raid alert, flag it, and count how often this happens.
  - **Non-target types as features:** only types that also exist in the test period can be features. After 2026-09-07 the only alert source is alerts.in.ua. It has artillery shelling and urban fighting, but no escalations or kamikaze-drone attacks. So escalations can be used for exploratory analysis and the write-up, but not as model inputs. Telegram threat posts (`@kpszsu`, `@deraketaua`) cover that information in both periods.

### Evaluation

- **Q9. Periods.**
  - Training start for the raion spec: from November 2025 only, or include earlier years with oblast-wide alerts spread to all raions plus a regime flag?
  - Validation period?
  - Test period: proposed as the last 4 weeks, about 2026-09-14 → 2026-10-11.
  - How often to retrain inside the backtest: weekly?
- **Q10. Which horizons to train:** all of 1–24, or a subset (1, 2, 3, 6, 12, 24) to cut compute?

### Features

- **Q11. Neighbour map.**
  - Source of raion boundaries, e.g. the admin-2 geojson.
  - Definition of a neighbour: units that share a border.
- **Q12. Telegram features.**
  - Final channel list.
  - Matching text to units: a list of oblast, raion and city names in all grammatical cases, mapped to raions.
  - Feature types: threat posts per unit in the last 1/3/6 h, threat type (Shahed, jet drone, ballistic, cruise missile, MiG-31K, bombers).
  - Edited posts: if a post was edited at or after t, use the original or drop it?
  - From which date the Telegram model is trained, based on a plot of posts per month.
  - Exclude LLM-based extraction? A model trained on data from the test period could leak.
- **Q13. One shared model or one per unit:** decided by an experiment on the validation period. Default: shared.

### Product

- **Q14. Deliverable form.**
  - CLI + README + a results notebook?
  - Plots as PNG or as an HTML report?
