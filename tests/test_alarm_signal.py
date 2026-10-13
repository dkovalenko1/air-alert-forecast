from airalert.alarm_signal import (CONTINUE, END, OTHER, RETRACT, START, UNKNOWN, Location, classify, classify_all,
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
    assert classify("🟢 Львівська обл.\n15:57 | Відбій тривоги. Будьте обережні!").kind == END  # time in front


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


def test_only_air_raid_apologies_are_retractions():
    for text in [
        "🟢 Шепетівка (Хмельницька обл.)\nПросимо вибачення, сталась технічна помилка. Загроза БПЛА в місті відсутня.",
        "🟢 Сумська обл.\nПросимо вибачення, сталась технічна помилка. Загроза обстрілу в області відсутня.",
        "📢 Київ\nПросимо вибачення за повторні звуки сирени перед відбоєм, сталась технічна помилка.",
        "🟢 Херсон (Херсонська обл.)\nПриносимо вибачення, сталася технічна помилка.\nВ Херсоні не було загрози артобстрілу.",
        "🟢 Київ\nПросимо вибачення, сталась технічна помилка. Загроза обстрілу в місті відсутня.\n"
        "Повітряна тривога в місті триває! Будьте обережні!",
        "🟢 Київ\nПросимо вибачення. Сталася помилка. Відбою не було!\nЗалишайтеся в укриттях!",
    ]:
        assert classify(text).kind == UNKNOWN, text
    assert classify("🟢 Київ\nПросимо вибачення, сталась технічна помилка. Тривога в місті відсутня.").kind == RETRACT
    assert classify("🟢 Сумська обл.\nПросимо вибачення, сталась технічна помилка. Повітряна тривога в області відсутня.").kind == RETRACT


def test_retraction_wordings_seen_in_the_channel():
    for text in [  # real wordings (message ids 25165, 48022, 57248, 149957 and others)
        "🟢 Дніпропетровська обл.\nПриносимо свої щирі вибачення. Сталася помилка щодо тривоги. Тривога в області відсутня.",
        "🟢 Світловодськ (Кіровоградська обл.)\nПриносимо вибачення, сталася технічна помилка.\nНа даний момент тривога у місті відсутня.",
        "📢 Славута (Хмельницька обл.)\nПросимо вибачення , сталась технічна помилка . Повітряна тривога у місті відсутня.",
        "🟢 Закарпатська обл.\nПросимо вибачення, сталась технічна помилка. Повітряна тривога в Закарпатській області відсутня.",
        "🟢 Сумська обл.\nПриносимо вибачення, сталася технічна помилка.\nВ області не було тривоги.",
        "🟢 Черкаська обл.\nПросимо вибачення, тривога була помилково оголошена Уманською міською радою.",
        "🟢 Луганська обл.\nТривога в області була помилковою, стався технічний збій — Дмитро Лунін.",
        "🟢 Київ\nВибачте, трапилась помилка. Повітряної тривоги немає.",
    ]:
        assert classify(text).kind == RETRACT, text


def test_apology_with_a_late_all_clear_ends_the_alert():
    for text in [
        "🟢 Київ\nПросимо вибачення, сталась технічна помилка. Відбій повітряної тривоги був о 22:29.",
        "🟢 Черкаська обл.\nПросимо вибачення, сталась технічна помилка. Відбій у Черкаському районі був о 18:48",
        "🟢 Харківська обл.\nПросимо вибачення, сталась технічна помилка. Станом на зараз відбій тривоги, з 02:26.",
    ]:
        assert classify(text).kind == END, text


def test_message_with_several_sections():
    text = ("🔴 Червоний рівень тривоги\n• Харків (Харківська обл.)\n\n"
            "🟢 Відбій тривоги\n• Звенигородський район (Черкаська обл.)\n• Уманський район (Черкаська обл.)")
    red, clear = classify_all(text)
    assert (red.kind, red.color, [l.name for l in red.locations]) == (START, "red", ["Харків"])
    assert (clear.kind, [l.name for l in clear.locations]) == (END, ["Звенигородський район", "Уманський район"])
    downgrade, clear = classify_all("🟡 Відбій червоної тривоги\n• Чугуївський район (Харківська обл.)\n\n"
                                    "🟢 Відбій тривоги\n• Ізюмський район (Харківська обл.)")
    assert (downgrade.kind, downgrade.color) == (START, "yellow")  # red is over, yellow continues
    assert clear.kind == END and clear.locations[0].name == "Ізюмський район"


def test_blank_lines_in_ordinary_posts_do_not_split_them():
    text = "🟢 Київ\nПросимо вибачення, сталась технічна помилка. Тривога в місті відсутня.\n\nПовітряна тривога в Київській області триває!"
    assert [m.kind for m in classify_all(text)] == [RETRACT]
    downgrade = classify("🟡 Харківський район (Харківська обл.)\nВідбій червоної тривоги, жовта тривога ще триває. Будьте обережні!")
    assert (downgrade.kind, downgrade.color) == (START, "yellow")
