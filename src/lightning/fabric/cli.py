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
from argparse import Namespace
from typing import Any, Optional

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

_JSONARGPARSE_SIGNATURES_AVAILABLE = RequirementCache("jsonargparse[signatures]>=4.27.7")
_LIGHTNING_SDK_AVAILABLE = RequirementCache("lightning_sdk")

_SUPPORTED_ACCELERATORS = ("cpu", "gpu", "cuda", "mps", "tpu", "auto")


def _get_supported_strategies() -> list[str]:
    """Returns strategy choices from the registry, with the ones removed that are incompatible to be launched from the
    CLI or ones that require further configuration by the user."""
    available_strategies = STRATEGY_REGISTRY.available_strategies()
    excluded = r".*(spawn|fork|notebook|xla|tpu|offload).*"
    return [strategy for strategy in available_strategies if not re.match(excluded, strategy)]


if _JSONARGPARSE_SIGNATURES_AVAILABLE:
    from jsonargparse import ArgumentParser, register_unresolvable_import_paths

    # Align with pytorch CLI behavior
    register_unresolvable_import_paths(torch)  # Required until the upstream PyTorch issue is fixed

    try:
        from jsonargparse import set_parsing_settings

        set_parsing_settings(config_read_mode_fsspec_enabled=True)
    except ImportError:
        from jsonargparse import set_config_read_mode

        set_config_read_mode(fsspec_enabled=True)
else:
    locals()["ArgumentParser"] = object


class FabricCLI:
    """Lightning Fabric command-line tool."""

    def __init__(self, args: Optional[list[str]] = None, run: bool = True) -> None:
        self.parser = self.init_parser()
        self._add_subcommands(self.parser)
        self.config, self.unknown_args = self.parser.parse_known_args(args)

        if run:
            self.run()

    def init_parser(self) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(prog="lightning-fabric", description=self.__class__.__doc__)

    def _add_subcommands(self, parser: ArgumentParser) -> None:
        """Adds subcommands to the parser."""
        subparsers = parser.add_subparsers(dest="command", required=True)
        self.add_run_subcommand(subparsers)
        self.add_consolidate_subcommand(subparsers)

    def add_run_subcommand(self, subparsers: Any) -> None:
        """Adds the `run` subcommand to the parser."""
        parser = subparsers.add_parser("run", help="Run a Lightning Fabric script.")
        parser.add_argument(
            "script",
            type=str,
            help="Path to the Python script with the code to run. The script must contain a Fabric object.",
        )
        parser.add_argument(
            "--accelerator",
            choices=_SUPPORTED_ACCELERATORS,
            default=None,
            help="The hardware accelerator to run on.",
        )
        parser.add_argument(
            "--strategy",
            choices=_get_supported_strategies(),
            default=None,
            help="Strategy for how to run across multiple devices.",
        )
        parser.add_argument(
            "--devices",
            type=str,
            default="1",
            help=(
                "Number of devices to run on (int), which devices to run on (list or str), or 'auto'."
                " The value applies per node."
            ),
        )
        parser.add_argument(
            "--num-nodes",
            "--num_nodes",
            type=int,
            default=1,
            help="Number of machines (nodes) for distributed execution.",
        )
        parser.add_argument(
            "--node-rank",
            "--node_rank",
            type=int,
            default=0,
            help="The index of the machine (node) this command gets started on. Must be 0, ..., num_nodes - 1.",
        )
        parser.add_argument(
            "--main-address",
            "--main_address",
            type=str,
            default="127.0.0.1",
            help="The hostname or IP address of the main machine (usually the one with node_rank = 0).",
        )
        parser.add_argument(
            "--main-port",
            "--main_port",
            type=int,
            default=29400,
            help="The main port to connect to the main machine.",
        )
        parser.add_argument(
            "--precision",
            choices=list(get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_STR_ALIAS)),
            default=None,
            help=(
                "Double precision ('64-true' or '64'), full precision ('32-true' or '32'), "
                "half precision ('16-mixed' or '16') or bfloat16 precision ('bf16-mixed' or 'bf16')."
            ),
        )

    def add_consolidate_subcommand(self, subparsers: Any) -> None:
        """Adds the `consolidate` subcommand to the parser."""
        parser = subparsers.add_parser(
            "consolidate", help="Convert a distributed/sharded checkpoint into a single file."
        )
        parser.add_argument(
            "checkpoint_folder",
            type=str,
            help="Path to the input checkpoint folder.",
        )
        parser.add_argument(
            "--output_file",
            type=str,
            default=None,
            help=(
                "Path to the file where the converted checkpoint should be saved. The file should not already exist."
                " If no path is provided, the file will be saved next to the input checkpoint folder with the same"
                " name and a '.consolidated' suffix."
            ),
        )

    def run(self) -> None:
        """Runs the subcommand."""
        if self.config.command == "run":
            self._run_script()
        elif self.config.command == "consolidate":
            self._consolidate_checkpoint()

    def _run_script(self) -> None:
        """Runs the script with the given arguments."""
        config = self.config.run
        if not (os.path.isfile(config.script) and os.access(config.script, os.R_OK)):
            raise SystemExit(f"Script not found or is not a readable file: {config.script}")

        args = Namespace(**vars(config))
        main(args=args, script_args=self.unknown_args)

    def _consolidate_checkpoint(self) -> None:
        """Consolidates the checkpoint."""
        config = self.config.consolidate
        if not os.path.isdir(config.checkpoint_folder):
            raise SystemExit(f"Checkpoint folder not found: {config.checkpoint_folder}")

        args = Namespace(**vars(config))
        processed_args = _process_cli_args(args)
        checkpoint = _load_distributed_checkpoint(processed_args.checkpoint_folder)
        torch.save(checkpoint, processed_args.output_file)


def _entrypoint() -> None:
    """The CLI entrypoint."""
    FabricCLI()


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


if __name__ == "__main__":
    if not _JSONARGPARSE_SIGNATURES_AVAILABLE:  # pragma: no cover
        _log.error(
            "To use the Lightning Fabric CLI, you must have 'jsonargparse[signatures]>=4.27.7' installed."
            " Install it by running: pip install -U 'jsonargparse[signatures]>=4.27.7'."
        )
        raise SystemExit(1)

    _entrypoint()
