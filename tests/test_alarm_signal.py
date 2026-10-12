from airalert.alarm_signal import (CONTINUE, END, OTHER, RETRACT, START, UNKNOWN, Location, classify,
                                   parse_location)

RAION = Location("raion", "Шепетівський район", "Хмельницька обл.")


def test_raion_start_one_location():
    m = classify("🚨 Шепетівський район (Хмельницька обл.)\nПовітряна тривога. Прямуйте в укриття!")
    assert (m.kind, m.color, m.locations) == (START, None, (RAION,))


def test_oblast_start_2023_wording():
    m = classify("🚨 Харківська обл.\nТривога - Усі в укриття")
    assert m.kind == START
    assert m.locations == (Location("oblast", "Харківська обл.", "Харківська обл."),)


def test_all_clear_both_wordings():
    assert classify("🟢 Київ\nВідбій тривоги - будьте обережні").kind == END
    assert classify("🟢 Куп'янський район (Харківська обл.)\nВідбій тривоги. Будьте обережні!").kind == END
    assert classify("🟢 Сумська обл.\nВідбій тривоги та загрози обстрілу - будьте обережні!").kind == END


def test_level_messages_with_several_raions():
    m = classify("🟡 Жовтий рівень тривоги\n• Вараський район (Рівненська обл.)\n• Сарненський район (Рівненська обл.)")
    assert (m.kind, m.color) == (START, "yellow")
    assert [loc.name for loc in m.locations] == ["Вараський район", "Сарненський район"]
    assert classify("🔴 Синельниківський район (Дніпропетровська обл.)\nЧервоний рівень тривоги. Прямуйте в укриття!").color == "red"


def test_multi_location_all_clear_without_bullets():
    m = classify("🟢 Відбій тривоги\nБаштанський район (Миколаївська обл.)\nВознесенський район (Миколаївська обл.)")
    assert m.kind == END and len(m.locations) == 2


def test_city_zone_and_apostrophes():
    m = classify("🚨 Запоріжжя (Запорізька обл.)\nПовітряна тривога. Прямуйте в укриття!")
    assert m.locations == (Location("city", "Запоріжжя", "Запорізька обл."),)
    assert parse_location("🟢 Кам’янський район (Дніпропетровська обл.)").name == "Кам'янський район"


def test_channel_warnings_are_never_alerts():
    for text, threat in [
        ("🛸 Шепетівка (Хмельницька обл.)\nЗагроза застосування БПЛА. Перейдіть в укриття!", "drones"),
        ("💣 Суми (Сумська обл.)\nЗагроза застосування КАБів. Негайно прямуйте в укриття!", "guided_bombs"),
        ("🚀 Харків (Харківська обл.)\nЗагроза застосування високошвидкісних цілей. Негайно прямуйте в укриття!", "missiles"),
        ("💥 Херсон (Херсонська обл.)\nЗагроза обстрілу! Перейдіть в укриття!", "shelling"),
        ("⚠️ Слов'янськ (Донецька обл.)\nЗМІ повідомляють про вибухи. Будьте обережні!", "explosions_reported"),
        ("🟢 Нікополь (Дніпропетровська обл.)\nВідбій загрози обстрілу. Будьте обережні!", "shelling"),
    ]:
        m = classify(text)
        assert (m.kind, m.threat) == (OTHER, threat), text


def test_continuation_and_retraction():
    assert classify("📢 Київ\nТривога в місті триває! Будьте обережні!").kind == CONTINUE
    m = classify("🟢 Сумська обл.\nПросимо вибачення, сталась технічна помилка. Тривога в області відсутня.")
    assert m.kind == RETRACT


def test_announcements_and_summaries_are_unknown():
    assert classify("Привіт! Ми шукаємо відповідальних волонтерів у команду моніторингу єТривоги 🚨\n\nДеталі").kind == UNKNOWN
    assert classify("Мапа тривог станом на 21:00 – 12 жовтня:\n\n🔴 Червона тривога у 22 районах").kind == UNKNOWN
    assert classify(None).kind == UNKNOWN
