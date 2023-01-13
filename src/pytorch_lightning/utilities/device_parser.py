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
from typing import Any

from lightning_fabric.accelerators.cpu import _parse_cpu_cores as new_parse_cpu_cores
from lightning_fabric.accelerators.cuda import is_cuda_available as new_is_cuda_available
from lightning_fabric.accelerators.cuda import num_cuda_devices as new_num_cuda_devices
from lightning_fabric.accelerators.tpu import _parse_tpu_devices as new_parse_tpu_cores
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device as new_determine_root_gpu_device
from lightning_fabric.utilities.device_parser import _parse_gpu_ids as new_parse_gpu_ids
from pytorch_lightning.accelerators.hpu import _parse_hpus as new_parse_hpus
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


def parse_hpus(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.parse_hpus` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_parse_hpus(*args, **kwargs)


def determine_root_gpu_device(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.determine_root_gpu_device` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_determine_root_gpu_device(*args, **kwargs)


def is_cuda_available() -> bool:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.is_cuda_available` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. Please use `lightning_fabric.accelerators.cuda.is_cuda_available` instead."
    )
    return new_is_cuda_available()


def num_cuda_devices() -> int:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.num_cuda_devices` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. Please use `lightning_fabric.accelerators.cuda.num_cuda_devices` instead."
    )
    return new_num_cuda_devices()


def parse_cpu_cores(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.parse_cpu_cores` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_parse_cpu_cores(*args, **kwargs)


def parse_gpu_ids(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.parse_gpu_ids` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_parse_gpu_ids(*args, **kwargs)


def parse_tpu_cores(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.device_parser.parse_tpu_cores` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_parse_tpu_cores(*args, **kwargs)
