"""Command-line entry point: `uv run python -m airalert <command>`."""

import argparse

from airalert import backtest, labels, telegram, validation
from airalert.models import MODELS


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
    bt = commands.add_parser("backtest", help="walk-forward backtest on the validation or test period")
    bt.add_argument("period", choices=list(backtest.PERIODS))
    bt.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    commands.add_parser("check-labels", help="compare labels with the official channel (Q20) and alerts.in.ua (Q2)")
    args = parser.parse_args()

    if args.command == "telegram-login":
        telegram.login()
    elif args.command == "download-telegram":
        for channel in args.channels:
            telegram.download(channel)
    elif args.command == "check-labels":
        validation.main()
    elif args.command == "backtest":
        backtest.main(args.period, args.models)
    elif args.command == "build-labels":
        for channel in args.channels:
            labels.build(channel)


if __name__ == "__main__":
    main()
