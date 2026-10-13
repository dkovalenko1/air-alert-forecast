"""Command-line entry point: `uv run python -m airalert <command>`."""

import argparse

from airalert import backtest, experiments, labels, telegram, threats, validation
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
    commands.add_parser("build-telegram", help="turn @kpszsu and @deraketaua posts into dated threat events (Q12)")
    experiment = commands.add_parser("experiment", help="validation experiments E1-E3 (DESIGN.md section 3)")
    experiment.add_argument("name", choices=list(experiments.EXPERIMENTS))
    experiment.add_argument("--assume-e2", choices=["shared", "per unit"],
                            help="start E3 while E2 is still running, assuming its LightGBM choice (checked afterwards)")
    args = parser.parse_args()

    if args.command == "telegram-login":
        telegram.login()
    elif args.command == "download-telegram":
        for channel in args.channels:
            telegram.download(channel)
    elif args.command == "check-labels":
        validation.main()
    elif args.command == "build-telegram":
        threats.build()
    elif args.command == "experiment":
        if args.name == "lightgbm":
            experiments.lightgbm_settings(args.assume_e2)
        else:
            experiments.EXPERIMENTS[args.name]()
    elif args.command == "backtest":
        backtest.main(args.period, args.models)
    elif args.command == "build-labels":
        for channel in args.channels:
            labels.build(channel)


if __name__ == "__main__":
    main()
