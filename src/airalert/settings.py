"""Project paths and secrets read from .env."""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
# Telethon appends ".session"; the file is git-ignored because it is a login.
TELEGRAM_SESSION = ROOT / "airalert"


def secret(name: str) -> str:
    """Return a value from the environment or .env, or stop with a clear message."""
    value = os.environ.get(name)
    env_file = ROOT / ".env"
    if not value and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, raw = line.partition("=")
            if sep and key.strip() == name:
                value = raw.strip().strip("\"'")
    if not value:
        sys.exit(f"{name} is not set: add it to .env")
    return value
