"""Prepare or serve the RoboTwin policy."""

import argparse
import logging
import os
import sys
from pathlib import Path
from a2meas import PROJECT_ROOT, RESOURCE_ROOT, runtime_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--resources", type=Path, default=RESOURCE_ROOT)
    prepare_parser.add_argument("--sam", action="store_true", help="Prepare SAM for RSI perception")
    commands.add_parser("serve", add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "prepare":
        if remaining:
            parser.error(f"Unrecognized arguments: {' '.join(remaining)}")
        from .resources import prepare

        with runtime_environment():
            count = prepare((PROJECT_ROOT / args.resources).resolve(), sam=args.sam)
        print(f"Verified {count} resource files")
        return
    from vla_eval.model_servers.serve import run_server
    from .server import RoboTwinModelServer
    from .rsi.server import RoboTwinRSIServer

    sys.argv = [sys.argv[0], *remaining]
    logging.basicConfig(level=logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)
    with runtime_environment():
        run_server(
            RoboTwinRSIServer
            if os.environ.get("ROBOTWIN_RSI", "false") == "true"
            else RoboTwinModelServer
        )
