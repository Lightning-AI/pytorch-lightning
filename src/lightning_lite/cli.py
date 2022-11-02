# Copyright The PyTorch Lightning team.
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
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

from lightning_lite.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning_lite.utilities.device_parser import _parse_gpu_ids
from lightning_lite.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_13

_log = logging.getLogger(__name__)

_SUPPORTED_ACCELERATORS = ("cpu", "gpu", "cuda", "mps", "tpu")
_SUPPORTED_STRATEGIES = (None, "ddp", "dp", "deepspeed")
_SUPPORTED_PRECISION = ("64", "32", "16", "bf16")


def _parse_args() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser(description="Launch your script with the Lightning Lite CLI.")
    parser.add_argument("script", type=str, help="Path to the Python script with Lightning Lite inside.")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=_SUPPORTED_ACCELERATORS,
        help="The hardware accelerator to run on.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=_SUPPORTED_STRATEGIES,
        help="Strategy for how to run across multiple devices.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help=(
            "Number of devices to run on (``int``), which devices to run on (``list`` or ``str``), or ``'auto'``."
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
        help=(
            "The index of the machine (node) this command gets started on. Must be a number in the range"
            " 0, ..., num_nodes - 1."
        ),
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
        type=str,
        default="32",
        choices=_SUPPORTED_PRECISION,
        help=(
            "Double precision (``64``), full precision (``32``), half precision (``16``) or bfloat16 precision"
            " (``'bf16'``)"
        ),
    )

    args, script_args = parser.parse_known_args()
    return args, script_args


def _set_env_variables(args: Namespace) -> None:
    """Set the environment variables for the new processes.

    The Lite connector will parse the arguments set here.
    """
    os.environ["LT_CLI_USED"] = "1"
    os.environ["LT_ACCELERATOR"] = str(args.accelerator)
    if args.strategy is not None:
        os.environ["LT_STRATEGY"] = str(args.strategy)
    os.environ["LT_DEVICES"] = str(args.devices)
    os.environ["LT_NUM_NODES"] = str(args.num_nodes)
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

    if _IS_WINDOWS and _TORCH_GREATER_EQUAL_1_13:
        # TODO: remove once import issue is resolved: https://github.com/pytorch/pytorch/issues/85427
        _log.error(
            "On the Windows platform, this launcher is currently only supported on torch < 1.13 due to a bug"
            " upstream: https://github.com/pytorch/pytorch/issues/85427"
        )
        exit(1)

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


def main() -> None:
    args, script_args = _parse_args()
    _set_env_variables(args)
    _torchrun_launch(args, script_args)


if __name__ == "__main__":
    main()
