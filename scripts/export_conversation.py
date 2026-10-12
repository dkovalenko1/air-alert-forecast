"""Export the AI conversation to docs/ai-conversation.md (Q14).

The conversation ran in the T3 Code desktop app with Claude Opus 5.5 (model id claude-opus-5-5) at
xhigh reasoning effort. T3 Code runs Claude through the Claude Code agent harness, which writes the
session logs read here (~/.claude/projects/<project>/*.jsonl).

Every user message, assistant reply, tool call and tool result is kept, in order. Left out: the model's
internal reasoning ("thinking") and the hidden harness context (<system-reminder> blocks), which are not
messages. Values from .env (API tokens) are replaced with <redacted>; the script fails if any remain.
Very long tool outputs are shortened to their first and last part, marked as such; pass
--full-tool-output to keep them whole.

Usage: uv run python scripts/export_conversation.py [--full-tool-output]
"""

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = pathlib.Path.home() / ".claude" / "projects" / ("-" + str(ROOT).strip("/").replace("/", "-").replace("_", "-"))
OUT = ROOT / "docs" / "ai-conversation.md"
TOOL_OUTPUT_LIMIT = 6000


def secrets() -> list[str]:
    env = ROOT / ".env"
    if not env.exists():
        return []
    values = [line.partition("=")[2].strip().strip("\"'") for line in env.read_text().splitlines() if "=" in line]
    return [v for v in values if len(v) >= 6]


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


def records():
    rows = []
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    rows = [r for r in rows if r.get("type") in ("user", "assistant") and not r.get("isSidechain")]
    return sorted(rows, key=lambda r: r["timestamp"])


def main(full_tool_output: bool) -> None:
    blocks = [
        "# AI conversation\n",
        "Full record of the conversation with the AI assistant while building this project.\n\n"
        "- **App:** T3 Code (desktop), which runs Claude through the Claude Code agent harness\n"
        "- **Model:** Claude Opus 5.5 (`claude-opus-5-5`), reasoning effort: xhigh\n\n"
        "Exported from the session logs by `scripts/export_conversation.py`. Every user message, assistant reply, tool call and tool "
        "result is included in order. Not included: the model's internal reasoning and hidden harness context. "
        "API tokens are redacted, and very long tool outputs are shortened (marked where it happens).\n",
    ]
    counts = {"user": 0, "assistant": 0, "tool call": 0, "tool result": 0}
    for r in records():
        time = r["timestamp"][:19].replace("T", " ") + " UTC"
        content = r["message"]["content"]
        if r["type"] == "user":
            if isinstance(content, list) and any(p.get("type") == "tool_result" for p in content):
                for part in content:
                    if part.get("type") == "tool_result":
                        result = shorten(text_of(part.get("content") or ""), full_tool_output)
                        error = " (error)" if part.get("is_error") else ""
                        blocks.append(f"<details><summary>Tool result{error}</summary>\n\n{fence(result)}\n\n</details>\n")
                        counts["tool result"] += 1
                continue
            text = without_harness_context(text_of(content))
            if not text:
                continue
            if r.get("isMeta"):
                blocks.append(f"<details><summary>System: skill instructions loaded ({time})</summary>\n\n{fence(text)}\n\n</details>\n")
            elif text.startswith("<task-notification>"):
                blocks.append(f"**System notification** ({time}): background task finished.\n\n{fence(text)}\n")
            else:
                blocks.append(f"---\n\n## User ({time})\n\n{text}\n")
                counts["user"] += 1
        else:
            for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                if part.get("type") == "text" and part["text"].strip():
                    blocks.append(f"### Assistant ({time})\n\n{part['text'].strip()}\n")
                    counts["assistant"] += 1
                elif part.get("type") == "tool_use":
                    blocks.append(f"<details><summary>Tool call: {part['name']}</summary>\n\n{tool_call_text(part)}\n\n</details>\n")
                    counts["tool call"] += 1
    document = "\n".join(blocks)
    for value in secrets():
        document = document.replace(value, "<redacted>")
    assert not any(value in document for value in secrets())
    OUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: " + ", ".join(f"{n} {k}s" for k, n in counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-tool-output", action="store_true")
    main(parser.parse_args().full_tool_output)
