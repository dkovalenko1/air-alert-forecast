"""The AI conversation export (scripts/export_conversation.py): no message dropped, no secret left, tool calls
listed by name only, and each handoff described by what actually happened."""

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_conversation.py"
spec = importlib.util.spec_from_file_location("export_conversation", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def record(kind, content, minute, account="first", **extra):
    return {"type": kind, "timestamp": f"2026-10-12T20:{minute:02d}:00Z", "message": {"content": content},
            "_account": account, **extra}


def body(document: str) -> str:
    """The document without its fixed header."""
    return document.split("notes mark where.\n", 1)[1]


def test_mixed_user_record_keeps_text_and_counts_the_tool_result():
    rows = [record("user", [{"type": "tool_result", "content": "ok"}, {"type": "text", "text": "and please also do X"}], 1)]
    document, counts = export.render(rows)
    assert "and please also do X" in document and counts == {"user": 1, "assistant": 0, "tool call": 0, "tool result": 1}


def test_tool_calls_are_listed_by_name_only():
    rows = [
        record("assistant", [{"type": "text", "text": "Checking."},
                             {"type": "tool_use", "name": "Bash", "input": {"command": "secret_command --flag"}},
                             {"type": "tool_use", "name": "mcp__t3-code__task_status", "input": {"taskId": "abc"}}], 1),
        record("user", [{"type": "tool_result", "content": "long output that must not appear"}], 2),
        record("assistant", [{"type": "text", "text": "Done."}], 3),
    ]
    document, counts = export.render(rows)
    assert "- Tool call: Bash\n- Tool call: mcp__t3-code__task_status\n" in document
    assert "secret_command" not in document and "long output" not in document and '"abc"' not in document
    assert document.index("Checking.") < document.index("Tool call: Bash") < document.index("Done.")
    assert counts == {"user": 0, "assistant": 2, "tool call": 2, "tool result": 1}


def test_usage_limit_retries_and_switch_to_another_account_become_notes():
    rows = [
        record("assistant", [{"type": "text", "text": "Checking the findings."}], 1),
        record("assistant", [{"type": "text", "text": "You've hit your session limit"}], 2, isApiErrorMessage=True),
        record("user", "Continue where you left off.", 3),
        record("assistant", [{"type": "text", "text": "You've hit your session limit"}], 4, isApiErrorMessage=True),
        record("user", "Context handoff (full_thread_summary): … long summary …\n\nUser message:\nContinue where you left off.",
               5, account="second"),
        record("assistant", [{"type": "text", "text": "Back at work."}], 6, account="second"),
    ]
    document, counts = export.render(rows)
    text = body(document)
    assert "session limit" not in text and "Continue where you left off" not in text and "long summary" not in text
    assert text.count("the first Claude Code account reached its usage limit") == 1
    assert text.count("**Context handoff**") == 1 and "moved this same chat to the second Claude Code account" in text
    assert counts["user"] == 0 and counts["assistant"] == 2


def test_each_usage_limit_episode_is_noted_once():
    limit = {"isApiErrorMessage": True}
    rows = [record("assistant", [{"type": "text", "text": "You've hit your session limit"}], m, **limit) for m in (1, 2)]
    rows += [record("assistant", [{"type": "text", "text": "Working again."}], 3)]
    rows += [record("assistant", [{"type": "text", "text": "You've hit your session limit"}], 4, account="second", **limit)]
    text = body(export.render(rows)[0])
    assert text.count("reached its usage limit") == 2 and "the second Claude Code account reached" in text


def test_message_redelivered_after_a_failed_send_is_kept():
    rows = [
        record("assistant", [{"type": "text", "text": "Started the download."}], 1),
        record("user", "Context handoff (delta_since_target_last_seen):\nProvider context handoff. …\n\n"
                       "User message:\nWhen will we pull the data?\nQ6: confirm", 2),
    ]
    document, counts = export.render(rows)
    assert "## User" in document and "When will we pull the data?\nQ6: confirm" in document and counts["user"] == 1
    assert "Provider context handoff" not in document and "re-delivered" in document and "moved this chat" not in document


def test_switch_back_to_the_first_account_is_described_as_such():
    """The handoff that returned the chat to the first account: retries and a task notification come first in the
    same record, and the delta summary must not be mistaken for a failed send."""
    notification = ("<task-notification>\n<task-id>bi3</task-id>\n<status>stopped</status>\n<summary>Background shell "
                    "command didn't finish before the previous session ended</summary>\n</task-notification>")
    rows = [
        record("assistant", [{"type": "text", "text": "Should I go on with step 1?"}], 1, account="second"),
        record("user", f"Continue where you left off.\nContinue where you left off.\n<system-reminder>{notification}"
                       "</system-reminder>\nContext handoff (delta_since_target_last_seen):\nProvider context handoff. …\n"
                       "[Historical user; …] I think 1 and 2 look ok\n\nUser message:\nLet's go with step 1.", 2),
    ]
    text = body(export.render(rows)[0])
    assert "moved this chat back to the first Claude Code account" in text and "re-delivered" not in text
    assert "## User (2026-10-12 20:02:00 UTC)\n\nLet's go with step 1.\n" in text
    assert "Historical user" not in text and "Continue where you left off" not in text and "<task-notification>" not in text


def test_switch_is_detected_when_a_notification_record_comes_first():
    """As in the real log: the notification is its own record, already on the new account, just before the handoff."""
    notification = "<task-notification>\n<status>stopped</status>\n<summary>Stopped early</summary>\n</task-notification>"
    rows = [
        record("assistant", [{"type": "text", "text": "Should I go on with step 1?"}], 1, account="second"),
        record("user", notification, 2),
        record("user", "Context handoff (delta_since_target_last_seen):\n…\n\nUser message:\nLet's go with step 1.", 2),
        record("assistant", [{"type": "text", "text": "Starting."}], 3),
        record("user", "Context handoff (delta_since_target_last_seen):\n…\n\nUser message:\nAnd this one?", 4),
    ]
    text = body(export.render(rows)[0])
    assert text.count("moved this chat back to the first Claude Code account") == 1  # the second handoff stays on it
    assert text.count("re-delivered") == 1 and "background task stopped: Stopped early" in text


def test_task_notification_shows_its_real_status_on_one_line():
    notification = ("<task-notification>\n<task-id>x</task-id>\n<status>completed</status>\n<summary>Background command "
                    "\"Run experiment E1\" completed (exit code 0)</summary>\n</task-notification>")
    text = body(export.render([record("user", notification, 1)])[0])
    assert text.strip() == ('**System notification** (2026-10-12 20:01:00 UTC): background task completed: '
                            'Background command "Run experiment E1" completed (exit code 0)')


def test_skill_instructions_become_a_one_line_note():
    text = body(export.render([record("user", "Base directory for this skill: …\n# Domain Modeling\n…", 1, isMeta=True)])[0])
    assert "Domain Modeling" not in text and "skill instructions loaded" in text


def test_secrets_set_only_in_the_environment_are_redacted(tmp_path):
    values = export.secrets(tmp_path / "missing.env", {"ALERTS_IN_UA_TOKEN": "tok_abcdef123456", "HOME": "/Users/x"})
    assert values == ["tok_abcdef123456"]
    assert export.redact("token tok_abcdef123456 here", values) == "token <redacted> here"


def test_secrets_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text('TELEGRAM_API_HASH="0123456789abcdef"\nOTHER=x\n')
    assert export.secrets(env, {}) == ["0123456789abcdef"]


def test_short_named_secret_is_redacted_but_short_other_settings_are_not(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEBUG=1\nTELEGRAM_API_HASH=abc\n")
    values = export.secrets(env, {"TELEGRAM_API_ID": "12345"})  # one short secret in .env, one only in the environment
    assert values == ["12345", "abc"]
    assert export.redact("TELEGRAM_API_ID=12345, hash abc, DEBUG=1", values) == "TELEGRAM_API_ID=<redacted>, hash <redacted>, DEBUG=1"


def test_accounts_are_named_by_their_config_folder(tmp_path):
    assert export.account_of(tmp_path / ".claude" / "projects" / "-x") == "first"
    assert export.account_of(tmp_path / ".claude-second" / "projects" / "-x") == "second"


def test_compaction_shows_the_command_and_a_note_not_the_summary():
    """The user's /compact: the command stays as their message; Claude Code's caveat, output and summary of the
    earlier conversation become one note (the summary is not the user's words)."""
    rows = [
        record("assistant", [{"type": "text", "text": "Should we accept E4?"}], 1),
        record("user", "<local-command-caveat>The command below was run directly in Claude Code…</local-command-caveat>", 2,
               isMeta=True),
        record("user", "<command-name>/compact</command-name>\n  <command-message>compact</command-message>\n"
                       "  <command-args></command-args>", 2),
        record("user", "This session is being continued from a previous conversation… Summary: secret plans", 4,
               isCompactSummary=True),
        record("user", "<local-command-stdout>Compacted </local-command-stdout>", 4),
        record("user", "I would say let's try b)", 5),
    ]
    document, counts = export.render(rows)
    text = body(document)
    assert "## User (2026-10-12 20:02:00 UTC)\n\n/compact\n" in text and "let's try b)" in text and counts["user"] == 2
    assert text.count("**Context compaction**") == 1 and "secret plans" not in text and "Summary:" not in text
    assert "Compacted" not in text and "local-command" not in text and "skill instructions" not in text
