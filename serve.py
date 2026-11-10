"""Prepare and serve A²-MEAS policies for BEHAVIOR or RoboTwin."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    for name in ("behavior", "robotwin"):
        commands.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.benchmark == "behavior":
        from a2meas.behavior.cli import main as run
    else:
        from a2meas.robotwin.cli import main as run
    return run(remaining)


if __name__ == "__main__":
    main()
