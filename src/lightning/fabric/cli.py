# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
import sys
from argparse import Namespace
from typing import Optional

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import get_args

from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS
from lightning.fabric.strategies import STRATEGY_REGISTRY
from lightning.fabric.utilities.consolidate_checkpoint import _process_cli_args
from lightning.fabric.utilities.device_parser import _parse_gpu_ids, _select_auto_accelerator
from lightning.fabric.utilities.distributed import _suggested_max_num_threads
from lightning.fabric.utilities.load import _load_distributed_checkpoint

_log = logging.getLogger(__name__)

_JSONARGPARSE_AVAILABLE = RequirementCache("jsonargparse")
_LIGHTNING_SDK_AVAILABLE = RequirementCache("lightning_sdk")

if _JSONARGPARSE_AVAILABLE:
    from jsonargparse import ArgumentParser

_SUPPORTED_ACCELERATORS = ("cpu", "gpu", "cuda", "mps", "tpu", "auto")


def _get_supported_strategies() -> list[str]:
    """Returns strategy choices from the registry, with the ones removed that are incompatible to be launched from the
    CLI or ones that require further configuration by the user."""
    available_strategies = STRATEGY_REGISTRY.available_strategies()
    excluded = r".*(spawn|fork|notebook|xla|tpu|offload).*"
    return [strategy for strategy in available_strategies if not re.match(excluded, strategy)]


def _build_parser() -> "ArgumentParser":
    """Build the jsonargparse-based CLI parser with subcommands."""
    if not _JSONARGPARSE_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "To use the Lightning Fabric CLI, you must have `jsonargparse` installed. "
            "Install it by running `pip install -U jsonargparse`."
        )

    parser = ArgumentParser(description="Lightning Fabric command line tool")
    subcommands = parser.add_subcommands()

    # run subcommand
    run_parser = ArgumentParser(description="Run a Lightning Fabric script.")
    run_parser.add_argument(
        "--accelerator",
        type=str,
        choices=_SUPPORTED_ACCELERATORS,
        default=None,
        help="The hardware accelerator to run on.",
    )
    run_parser.add_argument(
        "--strategy",
        type=str,
        choices=_get_supported_strategies(),
        default=None,
        help="Strategy for how to run across multiple devices.",
    )
    run_parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help=(
            "Number of devices to run on (int), which devices to run on (list or str), or 'auto'. "
            "The value applies per node."
        ),
    )
    run_parser.add_argument(
        "--num_nodes",
        "--num-nodes",
        type=int,
        default=1,
        help="Number of machines (nodes) for distributed execution.",
    )
    run_parser.add_argument(
        "--node_rank",
        "--node-rank",
        type=int,
        default=0,
        help=(
            "The index of the machine (node) this command gets started on. Must be a number in the range "
            "0, ..., num_nodes - 1."
        ),
    )
    run_parser.add_argument(
        "--main_address",
        "--main-address",
        type=str,
        default="127.0.0.1",
        help="The hostname or IP address of the main machine (usually the one with node_rank = 0).",
    )
    run_parser.add_argument(
        "--main_port",
        "--main-port",
        type=int,
        default=29400,
        help="The main port to connect to the main machine.",
    )
    run_parser.add_argument(
        "--precision",
        type=str,
        choices=list(get_args(_PRECISION_INPUT_STR)) + list(get_args(_PRECISION_INPUT_STR_ALIAS)),
        default=None,
        help=(
            "Double precision ('64-true' or '64'), full precision ('32-true' or '32'), "
            "half precision ('16-mixed' or '16') or bfloat16 precision ('bf16-mixed' or 'bf16')."
        ),
    )
    run_parser.add_argument(
        "script",
        type=str,
        help="Path to the Python script with the code to run. The script must contain a Fabric object.",
    )
    subcommands.add_subcommand("run", run_parser, help="Run a Lightning Fabric script")

    # consolidate subcommand
    con_parser = ArgumentParser(
        description="Convert a distributed/sharded checkpoint into a single file that can be loaded with torch.load()."
    )
    con_parser.add_argument(
        "checkpoint_folder",
        type=str,
        help="Path to the checkpoint folder to consolidate.",
    )
    con_parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Path to the file where the converted checkpoint should be saved. The file should not already exist. "
            "If not provided, the file will be saved next to the input checkpoint folder with the same name and a "
            "'.consolidated' suffix."
        ),
    )
    subcommands.add_subcommand("consolidate", con_parser, help="Consolidate a distributed checkpoint")

    return parser


def _set_env_variables(args: Namespace) -> None:
    """Set the environment variables for the new processes.

    The Fabric connector will parse the arguments set here.

    """
    os.environ["LT_CLI_USED"] = "1"
    if args.accelerator is not None:
        os.environ["LT_ACCELERATOR"] = str(args.accelerator)
    if args.strategy is not None:
        os.environ["LT_STRATEGY"] = str(args.strategy)
    os.environ["LT_DEVICES"] = str(args.devices)
    os.environ["LT_NUM_NODES"] = str(args.num_nodes)
    if args.precision is not None:
        os.environ["LT_PRECISION"] = str(args.precision)


def _get_num_processes(accelerator: str, devices: str) -> int:
    """Parse the `devices` argument to determine how many processes need to be launched on the current machine."""

    if accelerator == "auto" or accelerator is None:
        accelerator = _select_auto_accelerator()
    if devices == "auto":
        if accelerator == "cuda" or accelerator == "mps" or accelerator == "cpu":
            devices = "1"
        else:
            raise ValueError(f"Cannot default to '1' device for accelerator='{accelerator}'")
    if accelerator == "gpu":
        parsed_devices = _parse_gpu_ids(devices, include_cuda=True, include_mps=True)
    elif accelerator == "cuda":
        parsed_devices = CUDAAccelerator.parse_devices(devices)
    elif accelerator == "mps":
        parsed_devices = MPSAccelerator.parse_devices(devices)
    elif accelerator == "tpu":
        raise ValueError("Launching processes for TPU through the CLI is not supported.")
    else:
        return CPUAccelerator.parse_devices(devices)
    return len(parsed_devices) if parsed_devices is not None else 0


def _torchrun_launch(args: Namespace, script_args: list[str]) -> None:
    """This will invoke `torchrun` programmatically to launch the given script in new processes."""
    import torch.distributed.run as torchrun

    num_processes = 1 if args.strategy == "dp" else _get_num_processes(args.accelerator, args.devices)

    torchrun_args = [
        f"--nproc_per_node={num_processes}",
        f"--nnodes={args.num_nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.main_address}",
        f"--master_port={args.main_port}",
        args.script,
    ]
    torchrun_args.extend(script_args)

    # set a good default number of threads for OMP to avoid warnings being emitted to the user
    os.environ.setdefault("OMP_NUM_THREADS", str(_suggested_max_num_threads()))
    torchrun.main(torchrun_args)


def main(args: Namespace, script_args: Optional[list[str]] = None) -> None:
    _set_env_variables(args)
    _torchrun_launch(args, script_args or [])


def _run_command(cfg: Namespace, script_args: list[str]) -> None:
    """Execute the 'run' subcommand with the provided config and extra script args."""
    main(args=Namespace(**cfg), script_args=script_args)


def _consolidate_command(cfg: Namespace) -> None:
    """Execute the 'consolidate' subcommand with the provided config."""
    args = Namespace(checkpoint_folder=cfg.checkpoint_folder, output_file=cfg.output_file)
    config = _process_cli_args(args)
    checkpoint = _load_distributed_checkpoint(config.checkpoint_folder)
    torch.save(checkpoint, config.output_file)


def cli_main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the Fabric CLI using jsonargparse."""
    if not _JSONARGPARSE_AVAILABLE:  # pragma: no cover
        _log.error(
            "To use the Lightning Fabric CLI, you must have `jsonargparse` installed."
            " Install it by running `pip install -U jsonargparse`."
        )
        raise SystemExit(1)

    parser = _build_parser()
    # parse_known_args so that for 'run' we can forward unknown args to the user script
    cfg, unknown = parser.parse_known_args(argv)

    if not getattr(cfg, "subcommand", None):
        parser.print_help()
        return

    if cfg.subcommand == "run":
        # unknown contains the script's own args
        _run_command(cfg.run, unknown)
    elif cfg.subcommand == "consolidate":
        _consolidate_command(cfg.consolidate)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":
    cli_main(sys.argv[1:])
