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


def post(i, text, date, edited=None):
    return {"id": i, "date": date, "edit_date": edited, "text": text}


def test_signature_lines_name_no_places():
    events = events_from_posts([post(1, "📡Київщина – по шахедах чисто\n\n🍑 @deraketaua | Полтава | Київ",
                                     "2026-08-01T10:00:00+00:00")], "deraketaua")
    units = set(events.unit_id.dropna())
    assert units and all(u.startswith("Київська область/") for u in units)  # not Kyiv city, not Poltava


def test_each_line_keeps_its_own_threat_and_places():
    """A status list: drones for Kharkiv oblast (the second line inherits them), all-clear for Odesa oblast only."""
    events = events_from_posts([post(1, "⚠️3 шахеди на Харківщині:\n3 на Богодухів з півночі\n\n📡Одещина – чисто",
                                     "2026-08-01T10:00:00+00:00")], "deraketaua")
    local = events[events.scope != "national"]
    assert set(local[local.type == "drones"].unit_id.str.split("/").str[0]) == {"Харківська область"}
    assert "Харківська область/Богодухівський район" in set(local[(local.type == "drones") & (local.scope == "local")].unit_id)
    assert set(local[local.type == "clear"].unit_id.str.split("/").str[0]) == {"Одеська область"}
    assert set(events[events.scope == "national"].type) == {"drones", "clear"}


def test_weak_all_clear_words_count_only_on_short_lines():
    assert threat_types("🟩 Станом на зараз активності стратегічної авіації немає", "deraketaua") == {"clear"}
    long_line = ("❤️Друзі, канал веду сам, в мене немає десятки адмінів, і саме завдяки вам канал існує та працює майже "
                 "24/7, я намагаюсь давати оперативну інформацію про шахеди без затримок")
    assert threat_types(long_line, "deraketaua") == {"drones"}


def test_a_follow_up_counts_from_when_the_post_it_borrows_from_was_last_edited():
    posts = [post(1, "🏍 Реактивний БпЛА на Полтавщині.", "2026-08-01T10:00:00+00:00", "2026-08-01T10:20:00+00:00"),
             post(2, "Курсом на Павлоград.", "2026-08-01T10:05:00+00:00", "2026-08-01T10:05:05+00:00")]
    events = events_from_posts(posts, "kpszsu")
    assert (events[events.post_id == 2].available_at == pd.Timestamp("2026-08-01T10:20:00Z")).all()
