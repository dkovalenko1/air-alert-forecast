import json

import pandas as pd
import pytest

from airalert.validation import monthly_report


def alerts(rows):
    return pd.DataFrame([(u, pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC") if e else pd.NaT) for u, s, e in rows],
                        columns=["unit_id", "started_at", "finished_at"])


def test_events_match_within_five_minutes_in_both_directions():
    reference = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:40:00"),
                        ("A/a", "2026-08-02 10:00:00", "2026-08-02 11:00:00"),
                        ("A/b", "2026-08-03 10:00:00", "2026-08-03 10:30:00")])
    labels = alerts([("A/a", "2026-08-01 10:00:20", "2026-08-01 10:40:30"),  # 20-30 s later: found
                     ("A/a", "2026-08-02 10:07:00", "2026-08-02 11:00:10"),  # start 7 min later: not found
                     ("A/c", "2026-08-03 10:00:00", "2026-08-03 10:30:00")])  # another unit: not found
    report = monthly_report(labels, reference, "2026-08-01", "2026-09-01").set_index(["direction", "kind"])
    assert report.loc[("reference found in labels", "start"), "found"] == 1 / 3
    assert report.loc[("reference found in labels", "end"), "found"] == 2 / 3
    assert report.loc[("labels found in reference", "end"), "found"] == 2 / 3
    assert not report.passes.any()


def test_identical_sources_pass():
    same = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:40:00"), ("A/b", "2026-08-05 01:00:00", None)])
    assert monthly_report(same, same, "2026-08-01", "2026-09-01").passes.all()


def test_months_without_coverage_are_reported_and_fail():
    august = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:40:00")])
    report = monthly_report(august, august, "2026-07-01", "2026-10-01")
    assert sorted(report.month.unique()) == ["2026-07", "2026-08", "2026-09"] and len(report) == 12
    assert report[report.month == "2026-08"].passes.all()
    assert not report[report.month != "2026-08"].passes.any() and (report[report.month != "2026-08"].events == 0).all()
    empty = alerts([])
    assert not monthly_report(empty, empty, "2026-08-01", "2026-09-01").passes.any()  # no crash, no pass


def test_months_follow_kyiv_time_and_matches_cross_the_window_edge():
    late_evening = alerts([("A/a", "2026-08-31 22:10:00", "2026-08-31 22:50:00")])  # already September 1 in Kyiv
    assert monthly_report(late_evening, late_evening, "2026-08-01", "2026-09-01").events.sum() == 0
    reference = alerts([("A/a", "2026-08-31 20:59:00", "2026-08-31 21:30:00")])  # 23:59 Kyiv, inside August
    labels = alerts([("A/a", "2026-08-31 21:00:30", "2026-08-31 21:30:00")])  # 00:00:30 Kyiv, just outside
    report = monthly_report(labels, reference, "2026-08-01", "2026-09-01").set_index(["direction", "kind"])
    assert report.loc[("reference found in labels", "start"), "found"] == 1


def test_retracted_alerts_are_not_compared():
    labels = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:05:00"), ("A/a", "2026-08-02 10:00:00", "2026-08-02 11:00:00")])
    labels["retracted"] = [True, False]
    reference = alerts([("A/a", "2026-08-02 10:00:00", "2026-08-02 11:00:00")])
    assert monthly_report(labels, reference, "2026-08-01", "2026-09-01").passes.all()


def test_alerts_in_ua_kyiv_and_latest_snapshot(tmp_path, monkeypatch):
    from airalert import validation

    def snapshot(folder, finished_at):
        kyiv = {"id": 1, "alert_type": "air_raid", "location_type": "oblast", "location_title": "м. Київ",
                "location_oblast": "м. Київ", "started_at": "2026-10-12T18:25:14.086Z", "finished_at": finished_at}
        path = tmp_path / "alerts_in_ua" / folder / "31.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"alerts": [kyiv]}), encoding="utf-8")

    snapshot("2026-10-12T184322Z", None)
    snapshot("2026-10-12T193122Z", "2026-10-12T19:16:03.000Z")
    monkeypatch.setattr(validation, "RAW_DIR", tmp_path)
    frame = validation.alerts_in_ua_alerts()
    assert frame.unit_id.tolist() == ["м. Київ"]
    assert frame.finished_at.iloc[0] == pd.Timestamp("2026-10-12T19:16:03Z")


def test_utc_bounds_are_read_as_kyiv_months():
    late_evening = alerts([("A/a", "2026-08-31 22:10:00", "2026-08-31 22:50:00")])  # September 1 in Kyiv
    report = monthly_report(late_evening, late_evening, pd.Timestamp("2026-08-31 22:00", tz="UTC"), pd.Timestamp("2026-08-31 23:00", tz="UTC"))
    assert report.month.unique().tolist() == ["2026-09"] and report.passes.all()


def test_reference_outages_are_found_and_left_out():
    from airalert.validation import outages
    times = ["2026-08-29 04:00Z", "2026-08-29 04:55Z", "2026-08-30 12:42Z", "2026-09-02 13:48Z", "2026-09-02 14:00Z"]
    gaps = outages(times)
    assert gaps == [(pd.Timestamp("2026-08-29 04:55Z"), pd.Timestamp("2026-08-30 12:42Z")),
                    (pd.Timestamp("2026-08-30 12:42Z"), pd.Timestamp("2026-09-02 13:48Z"))]
    labels = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:40:00"), ("A/a", "2026-08-31 10:00:00", "2026-08-31 11:00:00")])
    reference = alerts([("A/a", "2026-08-01 10:00:00", "2026-08-01 10:40:00")])  # silent on 08-31
    assert not monthly_report(labels, reference, "2026-08-01", "2026-09-01").passes.all()
    assert monthly_report(labels, reference, "2026-08-01", "2026-09-01", exclude=gaps).passes.all()


def test_never_announced_alert_counts_its_all_clear_but_no_start():
    from airalert.validation import events
    labels = alerts([("A/a", "2026-08-01 05:00:00", "2026-08-01 10:00:00")]).assign(missed_start=[True])
    assert events(labels).kind.tolist() == ["end"]


def test_hourly_agreement_counts_unknown_hours_separately():
    from airalert.validation import hourly_agreement
    kyiv = "м. Київ"
    ours = alerts([(kyiv, "2026-08-01 10:00:00", "2026-08-01 10:30:00"), (kyiv, "2026-08-02 10:00:00", "2026-08-02 10:30:00")])
    reference = alerts([(kyiv, "2026-08-01 10:00:00", "2026-08-01 10:30:00"), (kyiv, "2026-08-03 10:00:00", "2026-08-03 10:30:00")])
    unknown = pd.DataFrame({"unit_id": [kyiv], "started_at": [pd.Timestamp("2026-08-03 09:00Z")], "finished_at": [pd.Timestamp("2026-08-03 12:00Z")],
                            "corrected_at": [pd.Timestamp("2026-08-03 12:00Z")], "unknown": [True]})
    [row] = hourly_agreement(ours, ours, unknown, reference, "2026-08-01", "2026-08-05", pd.Timestamp("2026-08-10", tz="UTC")).itertuples()
    kyiv_hours = 4 * 24
    assert row.unknown_share == pytest.approx(3 / (kyiv_hours * 119))  # three unknown hours, of all raion-hours
    assert row.ours_alert_reference_calm == pytest.approx(1 / (kyiv_hours * 119 - 3))  # 08-02 10:00
