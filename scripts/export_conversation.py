"""Export the AI conversation to docs/ai-conversation.md (Q14).

The conversation ran in the T3 Code desktop app with Claude Opus 5.5 (model id claude-opus-5-5) at
xhigh reasoning effort. T3 Code runs Claude through the Claude Code agent harness, which writes the
session logs read here (~/.claude*/projects/<project>/*.jsonl). When the first Claude Code account reached
its usage limit, T3 Code handed the same chat to a second account (a context handoff), whose logs live in its
own config folder (~/.claude-second); logs of every account are merged in time order.

Every user message, assistant reply, tool call and tool result is kept, in order. Left out: the model's
internal reasoning ("thinking") and the hidden harness context (<system-reminder> blocks), which are not
messages. The usage-limit replies, the "Continue where you left off." retries and the handoff summary T3 Code
gives the new account are replaced by two short notes saying what happened.

Secrets are replaced with <redacted>: every value in .env, and the project's secret variables
(SECRET_NAMES) also when they are set only in the environment; the script fails if any remain.
Very long tool outputs are shortened to their first and last part, marked as such; pass
--full-tool-output to keep them whole.

Usage: uv run python scripts/export_conversation.py [--full-tool-output]
"""

import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "-" + str(ROOT).strip("/").replace("/", "-").replace("_", "-")
LOG_DIRS = sorted(pathlib.Path.home().glob(f".claude*/projects/{PROJECT}"))  # one per Claude Code account
OUT = ROOT / "docs" / "ai-conversation.md"
TOOL_OUTPUT_LIMIT = 6000
SECRET_NAMES = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "ALERTS_IN_UA_TOKEN")  # read from .env or the environment
RETRY = "Continue where you left off."
HANDOFF = "Context handoff ("
LIMIT_NOTE = ("**Note** ({time}): the first Claude Code account reached its usage limit here, and the assistant's turn "
              "stopped. Retries failed until the context handoff below.\n")
HANDOFF_NOTES = {  # T3 Code wraps a message in a handoff; the summary is replaced by a note, the user's message is kept
    "full_thread_summary": "**Context handoff** ({time}): T3 Code moved this same chat to a second Claude Code account (a T3 "
                           "Code feature) and gave the new session a summary of the thread so far. The summary is not repeated "
                           "here: the full conversation is above. The conversation continues below.\n",
    "delta_since_target_last_seen": "**Context handoff** ({time}): the previous attempt to send the next message failed "
                                    "(background commands were still running), so T3 Code re-delivered it with a note of the "
                                    "failed attempt. The message follows.\n",
}


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
        elif part.get("type") == "tool_reference":
            parts.append(f"[tool loaded: {part.get('tool_name')}]")
    return "\n".join(parts)


def without_harness_context(text: str) -> str:
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()


def shorten(text: str, full: bool) -> str:
    if full or len(text) <= TOOL_OUTPUT_LIMIT:
        return text
    half = TOOL_OUTPUT_LIMIT // 2
    return f"{text[:half]}\n\n[… {len(text) - TOOL_OUTPUT_LIMIT:,} characters shortened in this export …]\n\n{text[-half:]}"


def fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{text}\n{ticks}"


def tool_call_text(part: dict) -> str:
    data = part.get("input", {})
    if part["name"] == "Bash" and "command" in data:
        head = f"{data.get('description', '')}\n\n" if data.get("description") else ""
        return head + fence(data["command"])
    if part["name"] in ("Write",) and "content" in data:
        return f"`{data.get('file_path')}`\n\n" + fence(data["content"])
    return fence(json.dumps(data, ensure_ascii=False, indent=2))


def records(log_dirs=LOG_DIRS):
    rows = []
    for path in sorted(p for d in log_dirs for p in d.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    rows = [r for r in rows if r.get("type") in ("user", "assistant") and not r.get("isSidechain")]
    return sorted(rows, key=lambda r: r["timestamp"])


def render(rows, full_tool_output: bool = False) -> tuple[str, dict]:
    blocks = [
        "# AI conversation\n",
        "Full record of the conversation with the AI assistant while building this project.\n\n"
        "- **App:** T3 Code (desktop), which runs Claude through the Claude Code agent harness\n"
        "- **Model:** Claude Opus 5.5 (`claude-opus-5-5`), reasoning effort: xhigh\n\n"
        "Exported from the session logs by `scripts/export_conversation.py`. Every user message, assistant reply, tool call and tool "
        "result is included in order. Not included: the model's internal reasoning and hidden harness context. "
        "API tokens are redacted, and very long tool outputs are shortened (marked where it happens). "
        "Midway, the chat moved to a second Claude Code account (a T3 Code context handoff); notes mark where.\n",
    ]
    counts = {"user": 0, "assistant": 0, "tool call": 0, "tool result": 0}
    limit_noted = False
    for r in rows:
        time = r["timestamp"][:19].replace("T", " ") + " UTC"
        content = r["message"]["content"]
        if r["type"] == "user":
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            # consecutive text/image parts form one message; tool results go in between, all in order
            groups = []
            for part in parts:
                if part.get("type") == "tool_result" or not groups or groups[-1][0] == "tool_result":
                    groups.append((part.get("type") if part.get("type") == "tool_result" else "message", [part]))
                else:
                    groups[-1][1].append(part)
            for kind, group in groups:
                if kind == "tool_result":
                    part = group[0]
                    result = shorten(text_of(part.get("content") or ""), full_tool_output)
                    error = " (error)" if part.get("is_error") else ""
                    blocks.append(f"<details><summary>Tool result{error}</summary>\n\n{fence(result)}\n\n</details>\n")
                    counts["tool result"] += 1
                    continue
                text = without_harness_context(text_of(group))
                if not text or text == RETRY:
                    continue
                if text.startswith(HANDOFF):
                    kind = text[len(HANDOFF):].partition(")")[0]
                    blocks.append(HANDOFF_NOTES.get(kind, HANDOFF_NOTES["full_thread_summary"]).format(time=time))
                    text = text.rpartition("\n\nUser message:\n")[2].strip() if "\n\nUser message:\n" in text else ""
                    if not text or text == RETRY:
                        continue
                if r.get("isMeta"):
                    blocks.append(f"<details><summary>System: skill instructions loaded ({time})</summary>\n\n{fence(text)}\n\n</details>\n")
                elif text.startswith("<task-notification>"):
                    blocks.append(f"**System notification** ({time}): background task finished.\n\n{fence(text)}\n")
                else:
                    blocks.append(f"---\n\n## User ({time})\n\n{text}\n")
                    counts["user"] += 1
        elif r.get("isApiErrorMessage"):  # "You've hit your session limit"
            if not limit_noted:
                blocks.append(LIMIT_NOTE.format(time=time))
                limit_noted = True
        else:
            for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                if part.get("type") == "text" and part["text"].strip():
                    blocks.append(f"### Assistant ({time})\n\n{part['text'].strip()}\n")
                    counts["assistant"] += 1
                elif part.get("type") == "tool_use":
                    blocks.append(f"<details><summary>Tool call: {part['name']}</summary>\n\n{tool_call_text(part)}\n\n</details>\n")
                    counts["tool call"] += 1
    return "\n".join(blocks), counts


def redact(document: str, values: list[str]) -> str:
    for value in values:
        document = document.replace(value, "<redacted>")
    assert not any(value in document for value in values)
    return document


def main(full_tool_output: bool) -> None:
    document, counts = render(records(), full_tool_output)
    OUT.write_text(redact(document, secrets()), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} from {len(LOG_DIRS)} log folders: " + ", ".join(f"{n} {k}s" for k, n in counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-tool-output", action="store_true")
    main(parser.parse_args().full_tool_output)
