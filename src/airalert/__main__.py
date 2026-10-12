"""Command-line entry point: `uv run python -m airalert <command>`."""

import argparse

from airalert import labels, telegram


def main() -> None:
    parser = argparse.ArgumentParser(prog="airalert", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("telegram-login", help="log in to Telegram once (asks for your phone number and a code)")
    download = commands.add_parser("download-telegram", help="download or update channel history (resumable)")
    download.add_argument("channels", nargs="*", default=list(telegram.CHANNELS),
                          help=f"channel usernames (default: {' '.join(telegram.CHANNELS)})")
    build = commands.add_parser("build-labels", help="turn downloaded channel messages into alert periods per unit")
    build.add_argument("channels", nargs="*", default=list(labels.CLASSIFIERS),
                       help=f"channels to parse (default: {' '.join(labels.CLASSIFIERS)})")
    args = parser.parse_args()

    if args.command == "telegram-login":
        telegram.login()
    elif args.command == "download-telegram":
        for channel in args.channels:
            telegram.download(channel)
    elif args.command == "build-labels":
        for channel in args.channels:
            labels.build(channel)


if __name__ == "__main__":
    main()
