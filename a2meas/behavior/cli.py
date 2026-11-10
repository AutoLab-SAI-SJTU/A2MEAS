"""Download weights and serve BEHAVIOR VLA policies."""

import argparse
import asyncio
import os
from pathlib import Path

from a2meas import RESOURCE_ROOT, cpu_layout, runtime_environment
from a2meas.behavior import DATA_ROOT, load_routes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--resources", type=Path, default=RESOURCE_ROOT)
    prepare.add_argument("--adapter", action="store_true", help="Prepare the twelve-task Adapter and SAM")
    serve = commands.add_parser("serve")
    serve.add_argument("--resources", type=Path, default=RESOURCE_ROOT)
    serve.add_argument("--gpu", type=int, default=0)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"],
                       default="max-autotune-no-cudagraphs")
    serve.add_argument("--adapter", action="store_true", help="Enable Adapter correction for the twelve configured tasks")
    serve.add_argument("--num-inference-steps", type=int, help="Override denoising steps for every task")
    serve.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)
    if args.command == "list":
        for task_id, route in load_routes()["tasks"].items():
            print(f"{task_id}: {route['task']}")
        return
    with runtime_environment():
        if args.command == "prepare":
            from a2meas.behavior.resources import prepare

            count = prepare(args.resources.resolve(), adapter=args.adapter)
            print(f"Verified {count} resource files")
        else:
            if args.num_inference_steps is not None and args.num_inference_steps < 1:
                parser.error("--num-inference-steps must be positive")
            if args.seed < 0:
                parser.error("--seed must be nonnegative")
            if args.gpu < 0 or args.workers < 1 or not 1 <= args.port <= args.port + args.workers - 1 <= 65535:
                parser.error("Use a nonnegative GPU index, positive worker count and ports in 1..65535")
            from a2meas.behavior.resources import verify_shared
            verify_shared(args.resources.resolve())
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            if args.adapter:
                os.environ["OMNIGIBSON_DATA_PATH"] = str(args.data_dir.resolve())
            from a2meas.behavior.server import RouteManager, run

            _, cpu_pools = cpu_layout(args.gpu, args.workers)
            managers = [RouteManager(args.resources.resolve(), cpus, adapter=args.adapter,
                                     num_inference_steps=args.num_inference_steps, seed=args.seed,
                                     compile_mode=args.compile_mode) for cpus in cpu_pools]

            async def serve_workers():
                await asyncio.gather(*(
                    run(manager, args.host, args.port + index)
                    for index, manager in enumerate(managers)
                ))

            try:
                asyncio.run(serve_workers())
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
