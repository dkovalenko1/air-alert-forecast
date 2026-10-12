from datetime import datetime, timedelta

from airalert.alarm_signal import END, ESCALATION, OTHER, START, UNKNOWN, Location
from airalert.alerts import build_alerts
from airalert.official_channel import classify
from airalert.units import resolve


def test_raion_start_and_all_clear():
    m = classify("🔴 09:08 Повітряна тривога в Ізюмський район\nСлідкуйте за подальшими повідомленнями.\n#Ізюмський_район")
    assert (m.kind, m.locations) == (START, (Location("raion", "Ізюмський район", None),))
    assert resolve(m.locations[0]) == ["Харківська область/Ізюмський район"]
    assert classify("🟢 12:33 Відбій тривоги в Ізюмський район.\nСлідкуйте.\n#Ізюмський_район").kind == END


def test_several_places_from_hashtags():
    m = classify("🟢 12:33 Відбій тривоги в\n• Мелітопольський район\n• Василівський район\n#Мелітопольський_район #Василівський_район")
    assert m.kind == END and [loc.name for loc in m.locations] == ["Мелітопольський район", "Василівський район"]


def test_hashtag_without_apostrophe_resolves():
    m = classify("🔴 10:00 Повітряна тривога в Кам'янський район\nСлідкуйте.\n#Камянський_район")
    assert resolve(m.locations[0]) == ["Дніпропетровська область/Кам'янський район"]


def test_kyiv_and_city_zones():
    assert resolve(classify("🔴 05:51 Повітряна тривога в м. Київ\nСлідкуйте.\n#м_Київ").locations[0]) == ["м. Київ"]
    kharkiv = classify("🔴 05:24 Повітряна тривога в м. Харків та Харківська територіальна громада\nСлідкуйте.\n"
                       "#м_Харків_та_Харківська_територіальна_громада")
    assert kharkiv.kind == START and resolve(kharkiv.locations[0]) == []


def test_partial_all_clear_closes_the_oblast_alert():
    m = classify("🟡 06:22 Відбій тривоги в Дніпропетровська область\n(тривога ще триває у якомусь районі)\n#Дніпропетровська_область")
    assert m.kind == END and m.locations[0].level == "oblast"


def test_escalations_and_other_threats():
    zap = "#м_Запоріжжя_та_Запорізька_територіальна_громада"
    assert classify(f"🔴 🔴 08:32 Загроза ударних БпЛА в м. Запоріжжя!\nУвага!\n{zap}").kind == ESCALATION
    assert classify(f"🔴 🔴 22:45 Загроза керованих авіабомб в м. Запоріжжя!\nУвага!\n{zap}").kind == ESCALATION
    assert classify(f"🔴 🔴 06:31 Ракетна загроза в м. Запоріжжя!\nУвага!\n{zap}").kind == ESCALATION
    nik = "#м_Нікополь_та_Нікопольська_територіальна_громада"
    assert classify(f"🔴 19:12 Загроза артобстрілу! Зараз у м. Нікополь\n{nik}").kind == OTHER
    assert classify(f"🟢 19:44 Відбій загрози артобстрілу в м. Нікополь.\n{nik}").kind == OTHER
    assert classify(f"🟢 08:00 Відбій загрози ударних БпЛА в м. Запоріжжя\n{zap}").kind == OTHER
    assert classify("🟢 Увага!\nБЕЗКОШТОВНА ЕВАКУАЦІЯ").kind == UNKNOWN


def test_escalation_inside_alert_keeps_its_times_and_without_alert_is_implied():
    t0 = datetime(2026, 9, 6, 21, 0)
    zap = "#м_Запоріжжя_та_Запорізька_територіальна_громада"
    msgs = [
        f"🔴 00:00 Повітряна тривога в м. Запоріжжя\nСлідкуйте.\n{zap}",
        f"🔴 🔴 00:49 Загроза ударних БпЛА в м. Запоріжжя!\nУвага!\n{zap}",
        f"🟢 02:48 Відбій тривоги в м. Запоріжжя.\nСлідкуйте.\n{zap}",
        f"🔴 🔴 03:10 Загроза керованих авіабомб в м. Запоріжжя!\nУвага!\n{zap}",
    ]
    minutes = [0, 49, 168, 190]
    h = build_alerts([(i, t0 + timedelta(minutes=m), classify(t)) for i, (t, m) in enumerate(zip(msgs, minutes))])
    first, second = h.alerts
    assert (first.started_at, first.finished_at, first.implied) == (t0, t0 + timedelta(minutes=168), False)
    assert second.implied and second.finished_at is None
