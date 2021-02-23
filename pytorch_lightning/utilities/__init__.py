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
"""General utilities"""

import numpy

from pytorch_lightning.utilities.apply_func import move_data_to_device  # noqa: F401
from pytorch_lightning.utilities.distributed import (  # noqa: F401
    AllGatherGrad,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)
from pytorch_lightning.utilities.enums import AMPType, DeviceType, DistributedType, LightningEnum  # noqa: F401
from pytorch_lightning.utilities.imports import (  # noqa: F401
    _APEX_AVAILABLE,
    _BOLTS_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _FAIRSCALE_PIPE_AVAILABLE,
    _GROUP_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _HYDRA_AVAILABLE,
    _HYDRA_EXPERIMENTAL_AVAILABLE,
    _module_available,
    _NATIVE_AMP_AVAILABLE,
    _OMEGACONF_AVAILABLE,
    _RPC_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_6,
    _TORCH_GREATER_EQUAL_1_7,
    _TORCH_LOWER_EQUAL_1_4,
    _TORCH_QUANTIZE_AVAILABLE,
    _TORCHTEXT_AVAILABLE,
    _TORCHVISION_AVAILABLE,
    _XLA_AVAILABLE,
)
from pytorch_lightning.utilities.parsing import AttributeDict, flatten_dict, is_picklable  # noqa: F401
from pytorch_lightning.utilities.xla_device import XLADeviceUtils  # noqa: F401

_TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps
