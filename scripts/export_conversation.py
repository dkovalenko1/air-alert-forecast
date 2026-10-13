"""Export the AI conversation to docs/ai-conversation.md (Q14).

The conversation ran in the T3 Code desktop app with Claude Opus 5.5 (model id claude-opus-5-5) at
xhigh reasoning effort. T3 Code runs Claude through the Claude Code agent harness, which writes the
session logs read here (~/.claude*/projects/<project>/*.jsonl). When one Claude Code account reached its usage
limit, T3 Code moved the same chat to another account (a context handoff); each account keeps its logs in its own
config folder (~/.claude for the first, ~/.claude-second for the second), and the logs are merged in time order.

Every user message and assistant reply is kept, in order. Tool calls are listed by name only ("Tool call: Bash");
their inputs and outputs are left out to keep the record readable. Also left out: the model's internal reasoning
("thinking"), the hidden harness context (<system-reminder> blocks) and the text of loaded skill instructions.
The usage-limit replies, the "Continue where you left off." retries and the summaries T3 Code gives an account at
a handoff are replaced by short notes saying what happened; background-task notifications become one line. A
`/compact` the user ran is kept as their message; the summary Claude Code then gave the model becomes a note.

Secrets are replaced with <redacted>: every value in .env, and the project's secret variables
(SECRET_NAMES) also when they are set only in the environment; the script fails if any remain.

Usage: uv run python scripts/export_conversation.py
"""

import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "-" + str(ROOT).strip("/").replace("/", "-").replace("_", "-")
LOG_DIRS = sorted(pathlib.Path.home().glob(f".claude*/projects/{PROJECT}"))  # one per Claude Code account
OUT = ROOT / "docs" / "ai-conversation.md"
SECRET_NAMES = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "ALERTS_IN_UA_TOKEN")  # read from .env or the environment
RETRY = "Continue where you left off."
HANDOFF = "Context handoff ("
FIRST_ACCOUNT = ".claude"  # the config folder of the account the chat started on


def account_of(log_dir: pathlib.Path) -> str:
    """'first' for ~/.claude, 'second' for any other account folder (~/.claude-second)."""
    return "first" if log_dir.parent.parent.name == FIRST_ACCOUNT else "second"


LIMIT_NOTE = ("**Note** ({time}): the {account} Claude Code account reached its usage limit here, and the assistant's "
              "turn stopped. Retries failed until the context handoff below.\n")
SWITCH_WITH_SUMMARY = ("**Context handoff** ({time}): T3 Code moved this same chat to the {account} Claude Code account "
                       "and gave the new session a summary of the thread so far. The summary is not repeated here: the "
                       "full conversation is above. The conversation continues below.\n")
SWITCH_WITH_DELTA = ("**Context handoff** ({time}): T3 Code moved this chat back to the {account} Claude Code account and "
                     "gave it the part of the thread it had not seen (the work above, done on the other account), which is "
                     "not repeated here. The next message follows.\n")
FAILED_SEND = ("**Context handoff** ({time}): the previous attempt to send the next message failed (background commands "
               "were still running), so T3 Code re-delivered it with a note of the failed attempt. The message follows.\n")
TASK_NOTE = "**System notification** ({time}): background task {status}: {summary}\n"
COMPACTED = ("**Context compaction** ({time}): after the user's `/compact`, Claude Code replaced the earlier conversation "
             "in the model's working context with a summary. The summary is not repeated here: the full conversation is "
             "above. The conversation continues below.\n")


def secrets(env_file: pathlib.Path = ROOT / ".env", environ=os.environ) -> list[str]:
    lines = env_file.read_text().splitlines() if env_file.exists() else []
    pairs = [(key.strip(), raw.strip().strip("\"'")) for key, _, raw in (line.partition("=") for line in lines if "=" in line)]
    # the project's named secrets are redacted whatever their length (an API id can be short); other .env values
    # only from 6 characters, so that a short setting like DEBUG=1 does not blank out every "1"
    values = [v for k, v in pairs if v and (k in SECRET_NAMES or len(v) >= 6)]
    names = set(SECRET_NAMES) | {k for k, _ in pairs}
    values += [environ[name] for name in names if environ.get(name) and (name in SECRET_NAMES or len(environ[name]) >= 6)]
    return sorted(set(values), key=len, reverse=True)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image":
            parts.append("[image]")
    return "\n".join(parts)


def without_harness_context(text: str) -> str:
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()


def task_note(text: str, time: str) -> str:
    field = lambda name: (re.search(rf"<{name}>(.*?)</{name}>", text, re.S) or [None, "unknown"])[1].strip()  # noqa: E731
    return TASK_NOTE.format(time=time, status=field("status"), summary=field("summary"))


def records(log_dirs=LOG_DIRS):
    rows = []
    for log_dir in log_dirs:
        for path in sorted(log_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                rows.extend({**json.loads(line), "_account": account_of(log_dir)} for line in f)
    rows = [r for r in rows if r.get("type") in ("user", "assistant") and not r.get("isSidechain")]
    return sorted(rows, key=lambda r: r["timestamp"])


def render(rows) -> tuple[str, dict]:
    blocks = [
        "# AI conversation\n",
        "Full record of the conversation with the AI assistant while building this project.\n\n"
        "- **App:** T3 Code (desktop), which runs Claude through the Claude Code agent harness\n"
        "- **Model:** Claude Opus 5.5 (`claude-opus-5-5`), reasoning effort: xhigh\n\n"
        "Exported from the session logs by `scripts/export_conversation.py`. Every user message and assistant reply is "
        "included in order; tool calls are listed by name only, without their inputs and outputs. Not included: the "
        "model's internal reasoning and hidden harness context. API tokens are redacted. The chat moved between two "
        "Claude Code accounts when one reached its usage limit (T3 Code context handoffs); notes mark where.\n",
    ]
    counts = {"user": 0, "assistant": 0, "tool call": 0, "tool result": 0}
    # the account of the last assistant record: a handoff on another account means the chat moved (a notification
    # record written just before the handoff is already on the new account, so it must not count)
    in_limit, assistant_account, tool_run = False, None, []

    def flush_tools():
        if tool_run:
            blocks.append("".join(f"- Tool call: {name}\n" for name in tool_run))
            tool_run.clear()

    for r in rows:
        time = r["timestamp"][:19].replace("T", " ") + " UTC"
        account = r.get("_account", "first")
        switched = assistant_account is not None and account != assistant_account
        if r["type"] == "assistant":
            assistant_account = account
        content = r["message"]["content"]
        if r["type"] == "user":
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            counts["tool result"] += sum(p.get("type") == "tool_result" for p in parts)
            text = without_harness_context(text_of([p for p in parts if p.get("type") != "tool_result"]))
            if not text or text == RETRY or "<local-command-caveat>" in text or "<local-command-stdout>" in text:
                continue  # Claude Code's own lines around a slash command the user ran
            flush_tools()
            if r.get("isCompactSummary"):
                blocks.append(COMPACTED.format(time=time))
                continue
            if r.get("isMeta"):
                blocks.append(f"**System** ({time}): skill instructions loaded (not repeated here).\n")
                continue
            if command := re.search(r"<command-name>(.*?)</command-name>", text, re.S):  # a slash command the user typed
                args = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
                text = f"{command.group(1).strip()} {args.group(1).strip() if args else ''}".strip()
            notification = re.search(r"<task-notification>.*?</task-notification>", text, re.S)
            if notification:
                blocks.append(task_note(notification.group(0), time))
                text = text.replace(notification.group(0), "").strip()
            if HANDOFF in text:
                kind = text[text.index(HANDOFF) + len(HANDOFF):].partition(")")[0]
                if kind == "full_thread_summary":
                    blocks.append(SWITCH_WITH_SUMMARY.format(time=time, account=account))
                elif switched:
                    blocks.append(SWITCH_WITH_DELTA.format(time=time, account=account))
                else:
                    blocks.append(FAILED_SEND.format(time=time))
                text = text.rpartition("\n\nUser message:\n")[2].strip() if "\n\nUser message:\n" in text else ""
            text = "\n".join(line for line in text.splitlines() if line.strip() != RETRY).strip()
            if text:
                blocks.append(f"---\n\n## User ({time})\n\n{text}\n")
                counts["user"] += 1
        elif r.get("isApiErrorMessage"):  # "You've hit your session limit"
            if not in_limit:
                flush_tools()
                blocks.append(LIMIT_NOTE.format(time=time, account=account))
                in_limit = True
        else:
            in_limit = False
            for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                if part.get("type") == "text" and part["text"].strip():
                    flush_tools()
                    blocks.append(f"### Assistant ({time})\n\n{part['text'].strip()}\n")
                    counts["assistant"] += 1
                elif part.get("type") == "tool_use":
                    tool_run.append(part["name"])
                    counts["tool call"] += 1
    flush_tools()
    return "\n".join(blocks), counts


def redact(document: str, values: list[str]) -> str:
    for value in values:
        document = document.replace(value, "<redacted>")
    assert not any(value in document for value in values)
    return document


def main() -> None:
    document, counts = render(records())
    OUT.write_text(redact(document, secrets()), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} from {len(LOG_DIRS)} log folders: " + ", ".join(f"{n} {k}s" for k, n in counts.items()))


if __name__ == "__main__":
    main()
