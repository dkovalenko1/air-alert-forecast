"""Command-line entry point: `uv run python -m airalert <command>`."""

import argparse
import pathlib

from airalert import backtest, experiments, final, forecast, labels, telegram, threats, validation
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
    bt = commands.add_parser("backtest", help="walk-forward backtest on the validation period (the test period: test-run)")
    bt.add_argument("period", choices=["validation"])
    bt.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    commands.add_parser("check-labels", help="compare labels with the official channel (Q20) and alerts.in.ua (Q2)")
    commands.add_parser("build-telegram", help="turn @kpszsu and @deraketaua posts into dated threat events (Q12)")
    experiment = commands.add_parser("experiment", help="validation experiments E1-E5 (DESIGN.md section 3)")
    experiment.add_argument("name", choices=list(experiments.EXPERIMENTS))
    experiment.add_argument("--assume-e2", choices=["shared", "per unit"],
                            help="start E3 while E2 is still running, assuming its LightGBM choice (checked afterwards)")
    test_run = commands.add_parser("test-run", help="the single test run with the configuration chosen on validation")
    test_report = commands.add_parser("test-report", help="intervals, weekly and per-raion scores, from the saved test predictions")
    for command in (test_run, test_report):
        command.add_argument("--out", type=pathlib.Path, default=final.REPORT,
                             help="report folder (default: reports/test; use another one to reproduce the run)")
    fit = commands.add_parser("train", help="fit the model chosen on validation on everything known at a cutoff, and save it")
    fit.add_argument("--cutoff", help="Kyiv time, e.g. '2026-10-12 21:00' (default: the last whole hour of the data)")
    pred = commands.add_parser("predict", help="print each unit's alert probability per horizon, with the saved model")
    pred.add_argument("--at", help="forecast time, Kyiv time (default: the last whole hour of the data)")
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
    elif args.command == "test-run":
        final.main(args.out)
    elif args.command == "test-report":
        final.report(args.out)
    elif args.command == "train":
        forecast.train(args.cutoff)
    elif args.command == "predict":
        forecast.predict(args.at)
    elif args.command == "backtest":
        backtest.main(args.period, args.models)
    elif args.command == "build-labels":
        for channel in args.channels:
            labels.build(channel)


if __name__ == "__main__":
    main()
