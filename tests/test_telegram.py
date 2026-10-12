import json

from airalert.telegram import _last_saved_id


def write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_empty_channel_starts_from_zero(tmp_path):
    assert _last_saved_id(tmp_path) == 0


def test_resumes_after_newest_month(tmp_path):
    write_lines(tmp_path / "2026-08.jsonl", [json.dumps({"id": 1}), json.dumps({"id": 2})])
    write_lines(tmp_path / "2026-09.jsonl", [json.dumps({"id": 3})])
    assert _last_saved_id(tmp_path) == 3


def test_drops_half_written_line_from_interrupted_run(tmp_path):
    newest = tmp_path / "2026-09.jsonl"
    newest.write_text(json.dumps({"id": 7}) + "\n" + '{"id": 8, "te', encoding="utf-8")
    assert _last_saved_id(tmp_path) == 7
    assert newest.read_text(encoding="utf-8") == json.dumps({"id": 7}) + "\n"


def test_falls_back_to_previous_month_if_newest_is_unreadable(tmp_path):
    write_lines(tmp_path / "2026-08.jsonl", [json.dumps({"id": 5})])
    (tmp_path / "2026-09.jsonl").write_text('{"id": 6, "te', encoding="utf-8")
    assert _last_saved_id(tmp_path) == 5
    assert not (tmp_path / "2026-09.jsonl").exists()
