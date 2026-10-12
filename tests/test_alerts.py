from datetime import datetime, timedelta

from airalert.alarm_signal import classify
from airalert.alerts import build_alerts

T0 = datetime(2026, 9, 6, 13, 0)
START = "🚨 Синельниківський район (Дніпропетровська обл.)\nПовітряна тривога. Прямуйте в укриття!"
RED = "🔴 Синельниківський район (Дніпропетровська обл.)\nЧервоний рівень тривоги. Прямуйте в укриття!"
YELLOW = "🟡 Жовтий рівень тривоги\n• Синельниківський район (Дніпропетровська обл.)\n• Павлоградський район (Дніпропетровська обл.)"
END = "🟢 Синельниківський район (Дніпропетровська обл.)\nВідбій тривоги. Будьте обережні!"


def run(*texts_and_minutes):
    return build_alerts([(i, T0 + timedelta(minutes=m), classify(t)) for i, (t, m) in enumerate(texts_and_minutes)])


def test_start_and_end_make_one_alert():
    h = run((START, 0), (END, 40))
    [a] = h.alerts
    assert (a.started_at, a.finished_at) == (T0, T0 + timedelta(minutes=40))


def test_repeated_start_keeps_original_start_and_level_rises_to_red():
    h = run((YELLOW, 0), (RED, 10), (END, 30))
    sinelnykove = [a for a in h.alerts if a.location.name == "Синельниківський район"]
    assert len(sinelnykove) == 1
    assert sinelnykove[0].started_at == T0 and sinelnykove[0].max_color == "red" and sinelnykove[0].restarts == 1


def test_locations_are_tracked_separately():
    h = run((YELLOW, 0), (END, 30))
    pavlohrad = [a for a in h.alerts if a.location.name == "Павлоградський район"]
    assert len(pavlohrad) == 1 and pavlohrad[0].finished_at is None  # still open: only Synelnykove got an all-clear


def test_alert_restarting_after_all_clear_is_a_new_alert():
    h = run((START, 0), (END, 18), (RED, 19))
    assert len(h.alerts) == 2  # alerts are kept exactly as published (Q7)


def test_retraction_ends_the_mistaken_alert_and_flags_it():
    retract = "🟢 Синельниківський район (Дніпропетровська обл.)\nПросимо вибачення, сталась технічна помилка. Тривога в області відсутня."
    h = run((START, 0), (retract, 2))
    [a] = h.alerts  # kept until the retraction (it was believed until then), flagged so labels can drop it
    assert a.retracted and a.finished_at == T0 + timedelta(minutes=2) and h.stats["retracted"] == 1


def test_all_clear_without_alert_is_counted_not_invented():
    h = run((END, 0))
    assert h.alerts == [] and h.stats["all-clear with no open alert"] == 1


def test_channel_warnings_do_not_open_alerts():
    drones = "🛸 Синельникове (Дніпропетровська обл.)\nЗагроза застосування БПЛА. Перейдіть в укриття!"
    assert run((drones, 0)).alerts == []
