import pandas as pd

from airalert.threats import events_from_posts, threat_types


def test_air_force_posts():
    assert threat_types("🛵 Сумщина: БпЛА в напрямку Кролевця.") == {"drones"}
    assert threat_types("🏍 Реактивний БпЛА на Полтавщині повз Решетилівку з півдня.") == {"jet_drones"}
    assert threat_types("☄️ Загроза застосування балістичного озброєння зі сходу!") == {"ballistic"}
    assert threat_types("💣КАБи ➡️ на Херсонщину") == {"guided_bombs"}
    assert threat_types("🚀КАБи на Донеччину!") == {"guided_bombs"}  # words win over the emoji
    assert threat_types("🛸 Ворожий розвідувальний БпЛА на сході від Комишувахи") == {"recon"}
    assert threat_types("🛫 Активність ворожої тактичної авіації в акваторії Чорного моря!") == {"aviation"}


def test_volunteer_posts():
    channel = "deraketaua"
    assert threat_types("⚠️2 реактивні шахеди з Чернігівщини на північ Київщини", channel) == {"jet_drones"}
    assert threat_types("🔴Загроза балістики з Брянська", channel) == {"ballistic"}
    assert threat_types("✈️В повітрі 4 стратегічні бомбардувальники Ту-22м3, курс в Чорне море", channel) == {"aviation"}
    assert threat_types("💥Пролунали вибухи у Києві", channel) == {"intercept"}


def test_all_clear_posts_are_only_clear():
    assert threat_types("📢Відбій загрози застосування балістичного озброєння!") == {"clear"}
    assert threat_types("📡Київщина – по шахедах чисто", "deraketaua") == {"clear"}
    assert threat_types("🟩 Станом на зараз активності бойової стратегічної авіації немає", "deraketaua") == {"clear"}


def test_posts_without_a_threat_have_no_type():
    assert threat_types("Слава Україні! Дякуємо нашим захисникам неба.") == frozenset()
    assert threat_types("") == frozenset() and threat_types(None) == frozenset()
    assert threat_types("🛵 що завгодно", "deraketaua") == frozenset()  # emoji fallback is for the Air Force only


def test_events_have_scopes_and_count_from_the_edit_time():
    posts = [{"id": 1, "date": "2026-08-01T10:00:00+00:00", "edit_date": "2026-08-01T10:00:09+00:00",
              "text": "🏍 Реактивний БпЛА курсом на Павлоград."},
             {"id": 2, "date": "2026-08-01T10:05:00+00:00", "edit_date": None, "text": "🛵 БпЛА на Сумщину."},
             {"id": 3, "date": "2026-08-01T10:06:00+00:00", "edit_date": None, "text": "Дякуємо!"}]
    events = events_from_posts(posts, "kpszsu")
    one = events[events.post_id == 1]
    assert set(one.scope) == {"national", "local"} and one.available_at.iloc[0] == pd.Timestamp("2026-08-01T10:00:09Z")
    assert one[one.scope == "local"].unit_id.tolist() == ["Дніпропетровська область/Павлоградський район"]
    two = events[events.post_id == 2]
    assert (two[two.scope == "oblast"].unit_id.str.startswith("Сумська область/")).sum() == 5
    assert 3 not in set(events.post_id)


def test_more_wording():
    assert threat_types("📡Не спостерігаються", "deraketaua") == {"clear"}
    assert threat_types("✈️Збили", "deraketaua") == {"intercept"}
    assert threat_types("Баражуючі боєприпаси на Чернігівщині") == {"drones"}


def test_follow_up_post_inherits_the_previous_type_within_30_minutes():
    posts = [{"id": 1, "date": "2026-08-01T10:00:00+00:00", "edit_date": None, "text": "🏍 Реактивний БпЛА на Полтавщині."},
             {"id": 2, "date": "2026-08-01T10:10:00+00:00", "edit_date": None, "text": "Курсом на Павлоград."},
             {"id": 3, "date": "2026-08-01T11:30:00+00:00", "edit_date": None, "text": "Курсом на Ізюм."},  # too late
             {"id": 4, "date": "2026-08-01T11:31:00+00:00", "edit_date": None, "text": "📢Відбій загрози!"},
             {"id": 5, "date": "2026-08-01T11:35:00+00:00", "edit_date": None, "text": "На Полтаву."}]  # after a clear
    events = events_from_posts(posts, "kpszsu")
    assert set(events[events.post_id == 2].type) == {"jet_drones"}
    assert not {3, 5} & set(events.post_id)
