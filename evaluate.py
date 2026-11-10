"""Evaluate A²-MEAS on BEHAVIOR or RoboTwin."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    for name in ("behavior", "robotwin"):
        commands.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.benchmark == "behavior":
        from benchmarks.behavior.evaluate import main as run

        return run(remaining)
    from benchmarks.robotwin.run import main as run

    return run(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
