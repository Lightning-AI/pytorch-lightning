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
from argparse import Namespace
from typing import Any, List, Optional

from lightning_utilities.core.imports import RequirementCache

from lightning_fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning_fabric.utilities.device_parser import _parse_gpu_ids

_log = logging.getLogger(__name__)

_CLICK_AVAILABLE = RequirementCache("click")

_SUPPORTED_ACCELERATORS = ("cpu", "gpu", "cuda", "mps", "tpu")
_SUPPORTED_STRATEGIES = ("ddp", "dp", "deepspeed")
_SUPPORTED_PRECISION = ("64", "32", "16", "bf16")

if _CLICK_AVAILABLE:
    import click

    @click.command(
        "model",
        context_settings=dict(
            ignore_unknown_options=True,
        ),
    )
    @click.argument(
        "script",
        type=click.Path(exists=True),
    )
    @click.option(
        "--accelerator",
        type=click.Choice(_SUPPORTED_ACCELERATORS),
        default=None,
        help="The hardware accelerator to run on.",
    )
    @click.option(
        "--strategy",
        type=click.Choice(_SUPPORTED_STRATEGIES),
        default=None,
        help="Strategy for how to run across multiple devices.",
    )
    @click.option(
        "--devices",
        type=str,
        default="1",
        help=(
            "Number of devices to run on (``int``), which devices to run on (``list`` or ``str``), or ``'auto'``."
            " The value applies per node."
        ),
    )
    @click.option(
        "--num-nodes",
        "--num_nodes",
        type=int,
        default=1,
        help="Number of machines (nodes) for distributed execution.",
    )
    @click.option(
        "--node-rank",
        "--node_rank",
        type=int,
        default=0,
        help=(
            "The index of the machine (node) this command gets started on. Must be a number in the range"
            " 0, ..., num_nodes - 1."
        ),
    )
    @click.option(
        "--main-address",
        "--main_address",
        type=str,
        default="127.0.0.1",
        help="The hostname or IP address of the main machine (usually the one with node_rank = 0).",
    )
    @click.option(
        "--main-port",
        "--main_port",
        type=int,
        default=29400,
        help="The main port to connect to the main machine.",
    )
    @click.option(
        "--precision",
        type=click.Choice(_SUPPORTED_PRECISION),
        default=None,
        help=(
            "Double precision (``64``), full precision (``32``), half precision (``16``) or bfloat16 precision"
            " (``'bf16'``)"
        ),
    )
    @click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
    def _run_model(**kwargs: Any) -> None:
        """Run a Lightning Fabric script.

        SCRIPT is the path to the Python script with the code to run. The script must contain a Fabric object.

        SCRIPT_ARGS are the remaining arguments that you can pass to the script itself and are expected to be parsed
        there.
        """
        script_args = list(kwargs.pop("script_args", []))
        main(args=Namespace(**kwargs), script_args=script_args)


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


def _torchrun_launch(args: Namespace, script_args: List[str]) -> None:
    """This will invoke `torchrun` programmatically to launch the given script in new processes."""
    import torch.distributed.run as torchrun

    if args.strategy == "dp":
        num_processes = 1
    else:
        num_processes = _get_num_processes(args.accelerator, args.devices)

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
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // num_processes)))
    torchrun.main(torchrun_args)


def main(args: Namespace, script_args: Optional[List[str]] = None) -> None:
    _set_env_variables(args)
    _torchrun_launch(args, script_args or [])


if __name__ == "__main__":
    if not _CLICK_AVAILABLE:  # pragma: no cover
        _log.error(
            "To use the Lightning Fabric CLI, you must have `click` installed."
            " Install it by running `pip install -U click`."
        )
        raise SystemExit(1)

    _run_model()
