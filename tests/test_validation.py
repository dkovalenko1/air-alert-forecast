import pandas as pd

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
