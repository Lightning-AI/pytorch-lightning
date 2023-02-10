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
"""General utilities."""

import numpy

from lightning_fabric.utilities import LightningEnum  # noqa: F401
from lightning_fabric.utilities import move_data_to_device  # noqa: F401
from pytorch_lightning.utilities.distributed import AllGatherGrad  # noqa: F401
from pytorch_lightning.utilities.enums import GradClipAlgorithmType  # noqa: F401
from pytorch_lightning.utilities.grads import grad_norm  # noqa: F401
from pytorch_lightning.utilities.imports import (  # noqa: F401
    _HIVEMIND_AVAILABLE,
    _OMEGACONF_AVAILABLE,
    _TORCH_QUANTIZE_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)
from pytorch_lightning.utilities.parameter_tying import find_shared_parameters, set_shared_parameters  # noqa: F401
from pytorch_lightning.utilities.parsing import AttributeDict, flatten_dict, is_picklable  # noqa: F401
from pytorch_lightning.utilities.rank_zero import (  # noqa: F401
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)

FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps
