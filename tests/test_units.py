import pytest

from airalert.alarm_signal import Location
from airalert.units import UnknownPlace, resolve, units


def test_registry_has_all_raions_and_kyiv():
    all_units = units()
    assert sum(u.kind == "raion" for u in all_units) == 122
    assert [u.unit_id for u in all_units if u.kind == "city"] == ["м. Київ"]
    assert {u.oblast for u in all_units if u.excluded} == {"Луганська область"}


def test_raion_by_current_and_old_name_and_apostrophe():
    assert resolve(Location("raion", "Самарівський район", "Дніпропетровська обл.")) == ["Дніпропетровська область/Самарівський район"]
    assert resolve(Location("raion", "Новомосковський район", "Дніпропетровська область")) == ["Дніпропетровська область/Самарівський район"]
    assert resolve(Location("raion", "Кам’янський район", "Дніпропетровська обл.")) == ["Дніпропетровська область/Кам'янський район"]


def test_oblast_alert_covers_all_its_raions_but_not_kyiv_city():
    covered = resolve(Location("oblast", "Київська обл.", "Київська обл."))
    assert len(covered) == 7 and "м. Київ" not in covered


def test_city_zones_map_to_no_unit_except_kyiv():
    assert resolve(Location("city", "Київ", None)) == ["м. Київ"]
    assert resolve(Location("city", "Запоріжжя", "Запорізька обл.")) == []


def test_unknown_names_are_reported():
    with pytest.raises(UnknownPlace):
        resolve(Location("raion", "Вигаданий район", "Сумська обл."))
