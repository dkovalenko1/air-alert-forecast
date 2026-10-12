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
    alerts, _, _ = unit_alerts_from_messages(messages((T("10:00"), OBLAST_START), (T("10:30"), OBLAST_END)), classify)
    assert len(alerts) == 5 and set(alerts.place_level) == {"oblast"}


def test_luhansk_is_excluded_and_unknown_places_are_counted():
    msgs = messages((T("10:00"), "🚨 Луганська обл.\nПовітряна тривога. Прямуйте в укриття!"),
                    (T("10:05"), "🚨 Вигаданий район (Сумська обл.)\nПовітряна тривога. Прямуйте в укриття!"))
    alerts, stats, _ = unit_alerts_from_messages(msgs, classify)
    assert alerts.empty and any(k.startswith("unknown place") for k in stats)


def test_overlapping_oblast_and_raion_alerts_merge_per_unit():
    msgs = messages((T("10:00"), OBLAST_START), (T("10:20"), RAION_START), (T("10:30"), OBLAST_END), (T("11:00"), RAION_END))
    alerts, _, _ = unit_alerts_from_messages(msgs, classify)
    merged = merge_per_unit(alerts, pd.Timestamp(T("12:00")))
    sumy = merged[merged.unit_id == "Сумська область/Сумський район"]
    assert len(sumy) == 1
    assert (sumy.started_at.iloc[0], sumy.finished_at.iloc[0]) == (pd.Timestamp(T("10:00")), pd.Timestamp(T("11:00")))
    other = merged[merged.unit_id == "Сумська область/Шосткинський район"]
    assert other.finished_at.iloc[0] == pd.Timestamp(T("10:30"))


def test_open_alert_runs_to_data_end_and_is_flagged():
    alerts, _, _ = unit_alerts_from_messages(messages((T("10:00"), RAION_START)), classify)
    merged = merge_per_unit(alerts, pd.Timestamp(T("12:00")))
    assert merged.finished_at.iloc[0] == pd.Timestamp(T("12:00")) and merged.open_at_end.iloc[0]


def test_start_under_old_raion_name_pairs_with_all_clear_under_new_name():
    msgs = messages((T("10:00"), "🚨 Новомосковський район (Дніпропетровська обл.)\nПовітряна тривога. Прямуйте в укриття!"),
                    (T("10:30"), "🟢 Самарівський район (Дніпропетровська обл.)\nВідбій тривоги. Будьте обережні!"))
    alerts, stats, _ = unit_alerts_from_messages(msgs, classify)
    assert len(alerts) == 1 and alerts.finished_at.iloc[0] == pd.Timestamp(T("10:30"))
    assert stats["all-clear with no open alert"] == 0


def test_every_message_is_counted_and_unrecognised_ones_are_returned():
    msgs = messages((T("10:00"), RAION_START), (T("10:10"), "Збираємо на дрони для бригади, посилання нижче"),
                    (T("10:20"), "🟢 Сумський район (Сумська обл.)\nВідбій чогось незрозумілого"), (T("10:30"), RAION_END))
    _, stats, unknown = unit_alerts_from_messages(msgs, classify)
    assert stats["message: start"] == 1 and stats["message: end"] == 1 and stats["message: unknown"] == 2
    assert [m["id"] for m in unknown] == [1, 2]


def test_retracted_alert_is_kept_flagged_and_left_out_of_label_periods():
    msgs = messages((T("10:00"), RAION_START),
                    (T("10:20"), "🟢 Сумський район (Сумська обл.)\nПросимо вибачення, сталась технічна помилка. Тривога в районі відсутня."))
    alerts, _, _ = unit_alerts_from_messages(msgs, classify)
    assert alerts.retracted.tolist() == [True] and alerts.finished_at.iloc[0] == pd.Timestamp(T("10:20"))
    assert merge_per_unit(alerts[~alerts.retracted], pd.Timestamp(T("12:00"))).empty


def test_messages_without_text_are_counted():
    msgs = [{"id": 1, "date": T("10:00"), "text": None, "service": "MessageActionChannelCreate"},
            {"id": 2, "date": T("10:01"), "text": "", "service": None, "media": "MessageMediaPhoto"}]
    _, stats, _ = unit_alerts_from_messages(msgs, classify)
    assert stats["message: service"] == 1 and stats["message: no text"] == 1


def test_all_clear_in_a_later_section_closes_the_alert():
    from airalert.alarm_signal import classify_all
    msgs = messages((T("10:00"), "🔴 Сумський район (Сумська обл.)\nЧервоний рівень тривоги. Прямуйте в укриття!"),
                    (T("10:40"), "🔴 Червоний рівень тривоги\n• Шосткинський район (Сумська обл.)\n\n🟢 Відбій тривоги\n• Сумський район (Сумська обл.)"))
    alerts, stats, _ = unit_alerts_from_messages(msgs, classify_all)
    sumy = alerts[alerts.unit_id == "Сумська область/Сумський район"]
    assert sumy.finished_at.tolist() == [pd.Timestamp(T("10:40"))] and stats["message with several sections"] == 1
    assert alerts[alerts.unit_id == "Сумська область/Шосткинський район"].started_at.tolist() == [pd.Timestamp(T("10:40"))]
