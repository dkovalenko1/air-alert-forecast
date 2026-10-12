import pandas as pd

from airalert.alarm_signal import classify
from airalert.labels import merge_per_unit, unit_alerts_from_messages

T = lambda hhmm: f"2026-09-06T{hhmm}:00+00:00"  # noqa: E731
OBLAST_START = "🚨 Сумська обл.\nПовітряна тривога. Прямуйте в укриття!"
OBLAST_END = "🟢 Сумська обл.\nВідбій тривоги. Будьте обережні!"
RAION_START = "🚨 Сумський район (Сумська обл.)\nПовітряна тривога. Прямуйте в укриття!"
RAION_END = "🟢 Сумський район (Сумська обл.)\nВідбій тривоги. Будьте обережні!"


def messages(*items):
    return [{"id": i, "date": d, "text": t} for i, (d, t) in enumerate(items)]


def test_oblast_alert_covers_every_raion_of_the_oblast():
    alerts, _ = unit_alerts_from_messages(messages((T("10:00"), OBLAST_START), (T("10:30"), OBLAST_END)), classify)
    assert len(alerts) == 5 and set(alerts.place_level) == {"oblast"}


def test_luhansk_is_excluded_and_unknown_places_are_counted():
    msgs = messages((T("10:00"), "🚨 Луганська обл.\nПовітряна тривога. Прямуйте в укриття!"),
                    (T("10:05"), "🚨 Вигаданий район (Сумська обл.)\nПовітряна тривога. Прямуйте в укриття!"))
    alerts, stats = unit_alerts_from_messages(msgs, classify)
    assert alerts.empty and any(k.startswith("unknown place") for k in stats)


def test_overlapping_oblast_and_raion_alerts_merge_per_unit():
    msgs = messages((T("10:00"), OBLAST_START), (T("10:20"), RAION_START), (T("10:30"), OBLAST_END), (T("11:00"), RAION_END))
    alerts, _ = unit_alerts_from_messages(msgs, classify)
    merged = merge_per_unit(alerts, pd.Timestamp(T("12:00")))
    sumy = merged[merged.unit_id == "Сумська область/Сумський район"]
    assert len(sumy) == 1
    assert (sumy.started_at.iloc[0], sumy.finished_at.iloc[0]) == (pd.Timestamp(T("10:00")), pd.Timestamp(T("11:00")))
    other = merged[merged.unit_id == "Сумська область/Шосткинський район"]
    assert other.finished_at.iloc[0] == pd.Timestamp(T("10:30"))


def test_open_alert_runs_to_data_end_and_is_flagged():
    alerts, _ = unit_alerts_from_messages(messages((T("10:00"), RAION_START)), classify)
    merged = merge_per_unit(alerts, pd.Timestamp(T("12:00")))
    assert merged.finished_at.iloc[0] == pd.Timestamp(T("12:00")) and merged.open_at_end.iloc[0]
