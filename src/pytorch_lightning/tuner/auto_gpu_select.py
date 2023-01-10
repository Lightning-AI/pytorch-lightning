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
from typing import List

import torch

from lightning_fabric.accelerators.cuda import num_cuda_devices
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


def pick_multiple_gpus(nb: int, _show_deprecation: bool = True) -> List[int]:
    """Pick a number of GPUs that are not yet in use.

    .. deprecated:: v1.9.0
        The function ``pick_multiple_gpus`` has been deprecated in v1.9.0 and will be removed in v2.0.0.
        Please use the function ``pytorch_lightning.accelerators.find_usable_cuda_devices`` instead.

    Raises:
        MisconfigurationException:
            If ``gpus`` or ``devices`` is set to 0, when ``auto_select_gpus=True``, or when the requested number is
            higher than the number of GPUs available on the machine.
    """
    if _show_deprecation:
        rank_zero_deprecation(
            "The function `pick_multiple_gpus` has been deprecated in v1.9.0 and will be removed in v2.0.0."
            " Please use the function `pytorch_lightning.accelerators.find_usable_cuda_devices` instead."
        )

    if nb == 0:
        raise MisconfigurationException(
            "auto_select_gpus=True, gpus=0 is not a valid configuration."
            " Please select a valid number of GPU resources when using auto_select_gpus."
        )

    num_gpus = num_cuda_devices()
    if nb > num_gpus:
        raise MisconfigurationException(f"You requested {nb} GPUs but your machine only has {num_gpus} GPUs.")
    nb = num_gpus if nb == -1 else nb

    picked: List[int] = []
    for _ in range(nb):
        picked.append(pick_single_gpu(exclude_gpus=picked, _show_deprecation=False))

    return picked


def pick_single_gpu(exclude_gpus: List[int], _show_deprecation: bool = True) -> int:
    """Find a GPU that is not yet in use.

    .. deprecated:: v1.9.0
        The function ``pick_single_gpu`` has been deprecated in v1.9.0 and will be removed in v2.0.0.
        Please use the function ``pytorch_lightning.accelerators.find_usable_cuda_devices`` instead.

    Raises:
        RuntimeError:
            If you try to allocate a GPU, when no GPUs are available.
    """
    if _show_deprecation:
        rank_zero_deprecation(
            "The function `pick_single_gpu` has been deprecated in v1.9.0 and will be removed in v2.0.0."
            " Please use the function `pytorch_lightning.accelerators.find_usable_cuda_devices` instead."
        )

    previously_used_gpus = []
    unused_gpus = []
    for i in range(num_cuda_devices()):
        if i in exclude_gpus:
            continue

        if torch.cuda.memory_reserved(f"cuda:{i}") > 0:
            previously_used_gpus.append(i)
        else:
            unused_gpus.append(i)

    # Prioritize previously used GPUs
    for i in previously_used_gpus + unused_gpus:
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
        except RuntimeError:
            continue
        return i
    raise RuntimeError("No GPUs available.")
