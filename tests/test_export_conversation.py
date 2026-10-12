"""The AI conversation export (scripts/export_conversation.py): nothing dropped, no secret left, handoff noted."""

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_conversation.py"
spec = importlib.util.spec_from_file_location("export_conversation", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def record(kind, content, minute, **extra):
    return {"type": kind, "timestamp": f"2026-10-12T20:{minute:02d}:00Z", "message": {"content": content}, **extra}


def test_mixed_user_record_keeps_text_and_tool_result():
    rows = [record("user", [{"type": "tool_result", "content": "ok"}, {"type": "text", "text": "and please also do X"}], 1)]
    document, counts = export.render(rows)
    assert "and please also do X" in document and counts == {"user": 1, "assistant": 0, "tool call": 0, "tool result": 1}


def test_usage_limit_retries_and_handoff_become_notes():
    rows = [
        record("assistant", [{"type": "text", "text": "Checking the findings."}], 1),
        record("assistant", [{"type": "text", "text": "You've hit your session limit"}], 2, isApiErrorMessage=True),
        record("user", "Continue where you left off.", 3),
        record("assistant", [{"type": "text", "text": "You've hit your session limit"}], 4, isApiErrorMessage=True),
        record("user", "Context handoff (full_thread_summary): … long summary …\n\nUser message:\nContinue where you left off.", 5),
        record("assistant", [{"type": "text", "text": "Back at work."}], 6),
    ]
    document, counts = export.render(rows)
    assert "session limit" not in document and "Continue where you left off" not in document and "long summary" not in document
    assert document.count("usage limit") == 1 and document.count("**Context handoff**") == 1
    assert counts["user"] == 0 and counts["assistant"] == 2


def test_secrets_set_only_in_the_environment_are_redacted(tmp_path):
    values = export.secrets(tmp_path / "missing.env", {"ALERTS_IN_UA_TOKEN": "tok_abcdef123456", "HOME": "/Users/x"})
    assert values == ["tok_abcdef123456"]
    assert export.redact("token tok_abcdef123456 here", values) == "token <redacted> here"


def test_secrets_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text('TELEGRAM_API_HASH="0123456789abcdef"\nOTHER=x\n')
    assert export.secrets(env, {}) == ["0123456789abcdef"]


def test_message_delivered_inside_a_handoff_is_kept():
    rows = [record("user", "Context handoff (delta_since_target_last_seen):\nProvider context handoff. …\n\n"
                           "User message:\nWhen will we pull the data?\nQ6: confirm", 1)]
    document, counts = export.render(rows)
    assert "## User" in document and "When will we pull the data?\nQ6: confirm" in document and counts["user"] == 1
    assert "Provider context handoff" not in document and "re-delivered" in document


def test_short_named_secret_is_redacted_but_short_other_settings_are_not(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEBUG=1\nTELEGRAM_API_HASH=abc\n")
    values = export.secrets(env, {"TELEGRAM_API_ID": "12345"})  # one short secret in .env, one only in the environment
    assert values == ["12345", "abc"]
    assert export.redact("TELEGRAM_API_ID=12345, hash abc, DEBUG=1", values) == "TELEGRAM_API_ID=<redacted>, hash <redacted>, DEBUG=1"
