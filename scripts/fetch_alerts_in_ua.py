"""Snapshot alerts.in.ua alert history before it rolls out of the API's one-month window.

Each response is saved exactly as received to data/raw/alerts_in_ua/<fetched_at>/<uid>.json.
Parsing happens later, so this script makes no decisions about the data.

Usage: python scripts/fetch_alerts_in_ua.py [--uids 4 8 9 ...]
Needs ALERTS_IN_UA_TOKEN in .env or in the environment.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HISTORY_URL = "https://api.alerts.in.ua/v1/regions/{uid}/alerts/month_ago.json"
# Every oblast-level location UID from the alerts.in.ua location table: 24 oblasts, Crimea (29),
# Sevastopol (30) and Kyiv city (31). Raw downloads are complete; exclusions happen when units are built.
OBLAST_UIDS = [3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
REQUEST_INTERVAL_S = 31  # the history endpoint allows 2 requests per minute
ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_token() -> str:
    token = os.environ.get("ALERTS_IN_UA_TOKEN")
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ALERTS_IN_UA_TOKEN":
                token = value.strip().strip("\"'")
    if not token:
        sys.exit("ALERTS_IN_UA_TOKEN is not set: add it to .env")
    return token


def fetch(uid: int, token: str) -> bytes:
    request = urllib.request.Request(
        HISTORY_URL.format(uid=uid),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 2:
                time.sleep(60)
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uids", type=int, nargs="+", default=OBLAST_UIDS)
    args = parser.parse_args()

    token = load_token()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = ROOT / "data" / "raw" / "alerts_in_ua" / fetched_at
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(args.uids)} locations, about {len(args.uids) * REQUEST_INTERVAL_S / 60:.0f} min "
          f"(the API allows 2 history requests per minute)", flush=True)

    failed = []
    for i, uid in enumerate(args.uids):
        if i:
            time.sleep(REQUEST_INTERVAL_S)
        try:
            body = fetch(uid, token)
        except urllib.error.HTTPError as error:
            print(f"[{i + 1}/{len(args.uids)}] uid {uid}: HTTP {error.code}", flush=True)
            failed.append(uid)
            continue
        (out_dir / f"{uid}.json").write_bytes(body)
        alerts = json.loads(body).get("alerts", [])
        print(f"[{i + 1}/{len(args.uids)}] uid {uid}: {len(alerts)} alerts", flush=True)

    print(f"Saved to {out_dir}" + (f"; failed uids: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
