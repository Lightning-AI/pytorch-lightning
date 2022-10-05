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

import os
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import torch.distributed.run as torchrun

from lightning_lite.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator


def _parse_args() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser()
    parser.add_argument("script", type=str, help="Path to the Python script with Lightning Lite inside.")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "cuda", "mps", "tpu"),
        help="The hardware accelerator to run on.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=(None, "ddp", "dp", "deepspeed"),
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
        type=int,
        default=1,
        help="Number of machines (nodes) for distributed execution.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help=(
            "The index of the machine (node) this command gets started on. Must be a number in the range"
            " 0, ..., num_nodes - 1."
        ),
    )
    parser.add_argument(
        "--main-address",
        type=str,
        default="127.0.0.1",
        help="The hostname or IP address of the main machine (usually the one with node_rank = 0).",
    )
    parser.add_argument(
        "--main-port",
        type=int,
        default=29400,
        help="The main port to connect to the main machine.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=("64", "32", "16", "bf16"),
        help=(
            "Double precision (``64``), full precision (``32``), half precision (``16``) or bfloat16 precision"
            " (``'bf16'``)"
        ),
    )

    args, script_args = parser.parse_known_args()
    return args, script_args


def _set_env_variables(args: Namespace) -> None:
    """Set the environment variables for the new processes. The Lite connector will parse the arguments set here."""
    os.environ["LT_ACCELERATOR"] = str(args.accelerator)
    if args.strategy is not None:
        os.environ["LT_STRATEGY"] = str(args.strategy)
    os.environ["LT_DEVICES"] = str(args.devices)
    os.environ["LT_NUM_NODES"] = str(args.num_nodes)
    os.environ["LT_PRECISION"] = str(args.precision)


def _get_num_processes(accelerator: str, devices: str) -> int:
    """Parse the `devices` argument to determine how many processes need to be launched on the current machine."""
    if accelerator in ("cuda", "gpu"):
        devices = CUDAAccelerator.parse_devices(devices)
    elif accelerator in ("mps", "gpu"):
        devices = MPSAccelerator.parse_devices(devices)
    elif accelerator == "tpu":
        raise ValueError("Launching processes for TPU through the CLI is not supported.")
    else:
        return CPUAccelerator.parse_devices(devices)
    return len(devices) if devices is not None else 0


def _torchrun_launch(args: Namespace, script_args: List[str]) -> None:
    """This will invoke `torchrun` programmatically to launch the given script in new processes."""
    if args.strategy == "dp":
        num_processes = 1
    else:
        num_processes = _get_num_processes(args.accelerator, args.devices)

    torchrun_args = []
    torchrun_args.extend(["--nproc_per_node", str(num_processes)])
    torchrun_args.extend(["--nnodes", str(args.num_nodes)])
    torchrun_args.extend(["--node_rank", str(args.node_rank)])
    torchrun_args.extend(["--master_addr", args.main_address])
    torchrun_args.extend(["--master_port", str(args.main_port)])
    torchrun_args.append(args.script)
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
