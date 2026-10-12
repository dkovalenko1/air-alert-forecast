from airalert.places import find_places


def names(text):
    return [(m.kind, m.name) for m in find_places(text)]


def test_air_force_style_posts():
    assert names("Реактивний БпЛА курсом на Павлоград.") == [("place", "Павлоград")]
    assert names("БпЛА на Харків з півночі.") == [("place", "Харків")]
    assert names("Курсом на Бучу/Ірпінь.") == [("place", "Буча"), ("place", "Ірпінь")]


def test_regions_and_cities_in_one_post():
    found = names("Група реактивних БпЛА через Полтавщину в напрямку Черкащини (Золотоноша).")
    assert found == [("oblast", "Полтавська область"), ("oblast", "Черкаська область"), ("place", "Золотоноша")]


def test_inflected_city_names():
    assert names("1 реактивний шахед з Хмельниччини на Тернопільщину в район Чорткова") == [
        ("oblast", "Хмельницька область"), ("oblast", "Тернопільська область"), ("place", "Чортків")]
    assert names("вибухи у Києві") == [("place", "Київ")]
    assert names("загроза для Білої Церкви") == [("place", "Біла Церква")]
    assert names("ракета на Кривий Ріг") == [("place", "Кривий Ріг")] == names("у Кривому Розі вибухи")


def test_raions_and_oblasts_by_name():
    assert names("Збиття в межах Ізюмського району") == [("raion", "Ізюмський район")]
    assert names("БпЛА у Харківській області") == [("oblast", "Харківська область")]


def test_oblast_mention_covers_all_its_raions():
    [m] = find_places("на Сумщину")
    assert len(m.unit_ids) == 5 and all(u.startswith("Сумська область/") for u in m.unit_ids)


def test_no_false_hits_on_common_words():
    assert names("12 березня нічна атака, загальна сума збитих цілей") == []
    assert names("Повітряні сили ЗСУ: обережно, бережіть себе") == []
