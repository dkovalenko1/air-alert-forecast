---
status: accepted
---

# Features see the alert history as published; labels see it as corrected. Edited messages keep their posting time

A forecast at t may only use what was known before t (ADR-0001). Two kinds of later information could break that:
corrections posted after an alert, and edits to a message after it was posted.

**Corrections (retractions).** When the label channel apologises for an alert posted by mistake ("Просимо
вибачення, сталась технічна помилка. Тривога в області відсутня."), the alert was still believed until the apology.
So two period tables are built from the same messages:
- **feature periods:** the retracted alert lasts from its start until the retraction;
- **label periods:** the retracted alert is dropped, because it never really happened.

The grid reads features (including "was the target hour on alert a day / a week ago") from feature periods only and
labels from label periods only, so a correction posted after t changes no feature at t.

Training labels follow the same rule. A model retrained at cutoff C sees each label as it was known at C: if a
retraction posted after C corrected a training hour, that hour keeps its published label. For this the grid also
holds, per horizon, the published label and when a correction changed it. Scoring uses the corrected labels.

The same split handles lost messages (Q17, Q20). After a missing all-clear, the alert stays in the feature periods
as published (nobody knew the all-clear was missing), while the stretch with no news becomes unknown in the labels.
An all-clear for an alert that was never announced adds nothing to the feature periods (it was never published),
while the hours before it become unknown in the labels. Unknown hours are neither trained on nor scored, and each
becomes unknown only from the moment it could have been detected.

An apology that gives a late all-clear ("…відбій повітряної тривоги був о 22:29") ends the alert when the apology is
posted, not at the earlier time it names, which was not known before the message.

**Edited messages.** Telegram returns only a message's latest text, dated at its original posting time, and keeps no
edit history. In the label channel every message has been edited at least once:
- two channel-wide re-edits, on 2024-01-30 (about 40,000 messages) and 2025-06-27 (about 187,000);
- since July 2025, another edit after posting: half within 29 seconds, but 13% more than 5 minutes later and 2.7%
  more than an hour later (July 2025 – October 2026);
- sampled late edits still carry the channel's fixed wording, the same as unedited messages (e.g. a start
  message edited 65 minutes later still reads "🚨 … район (… обл.) Повітряна тривога. Прямуйте в укриття!").

So edits are treated as not changing an alert's kind or place, and each message keeps its posting time. This is an
assumption backed by the evidence above, not a proof. The month-by-month comparison with the independent official
channel (Q20, 98% pass mark both ways; run 2026-10-13, see the README) is an approximate cross-check: an edit that moved an alert to
another place or changed its kind would mostly show up as a mismatch, but matching within 5 minutes cannot catch
every case (two short alerts close together can match one).

For the Telegram feature channels (`@kpszsu`, `@deraketaua`), whose posts are written by people and can change
meaning, the stricter rule from Q12 applies: a post counts only from its last edit (`threats.py`), so a post edited
at or after t is not used for forecasts at t.

## Considered options

- **Date every label-channel message by its edit time:** strictly safe, but the two re-edits would date about 227,000
  messages to 2024-01-30 or 2025-06-27, destroying the history.
- **Delete retracted alerts from the history (the first implementation):** labels are right, but a forecast before
  the retraction would "know" the alert was false. That is leakage; a code review found it.

## Consequences

- `labels.build` writes `<channel>_merged.parquet` (feature periods), `<channel>_label_periods.parquet` and
  `<channel>_corrections.parquet` (retracted alerts, with when they were retracted); the backtest passes all three
  to `build_grid`.
- After a retraction, history features still count the retracted alert's start (e.g. starts in the last 24 h).
  That is what was published, so it is not leakage, only slightly noisier.
- The README's leakage section states the edit assumption and the evidence.
