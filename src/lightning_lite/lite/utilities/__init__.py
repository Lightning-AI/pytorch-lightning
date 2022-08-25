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
"""General utilities."""

from lightning_lite.lite.utilities.apply_func import move_data_to_device  # noqa: F401
from lightning_lite.lite.utilities.distributed import AllGatherGrad  # noqa: F401
from lightning_lite.lite.utilities.enums import (  # noqa: F401
    _AcceleratorType,
    _StrategyType,
    AMPType,
    GradClipAlgorithmType,
    LightningEnum,
)
from lightning_lite.lite.utilities.imports import (  # noqa: F401
    _APEX_AVAILABLE,
    _FAIRSCALE_FULLY_SHARDED_AVAILABLE,
    _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE,
    _GROUP_AVAILABLE,
    _HIVEMIND_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _HPU_AVAILABLE,
    _IPU_AVAILABLE,
    _IS_INTERACTIVE,
    _IS_WINDOWS,
    _module_available,
    _OMEGACONF_AVAILABLE,
    _POPTORCH_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_10,
    _TORCH_GREATER_EQUAL_1_11,
    _TORCH_GREATER_EQUAL_1_12,
    _TORCH_QUANTIZE_AVAILABLE,
    _TORCHTEXT_AVAILABLE,
    _TORCHVISION_AVAILABLE,
    _TPU_AVAILABLE,
    _XLA_AVAILABLE,
)
from lightning_lite.lite.utilities.parameter_tying import find_shared_parameters, set_shared_parameters  # noqa: F401
from lightning_lite.lite.utilities.rank_zero import (  # noqa: F401
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)
