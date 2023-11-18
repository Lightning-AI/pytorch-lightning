# Copyright The Lightning AI team.
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

import torch
from lightning_utilities.core.imports import RequirementCache

_INDEX_FILENAME = "index.json"
_DEFAULT_CHUNK_BYTES = 1 << 26  # 64M B
_DEFAULT_FAST_DEV_RUN_ITEMS = 10

# This is required for full pytree serialization / deserialization support
_TORCH_GREATER_EQUAL_2_1_0 = RequirementCache("torch>=2.1.0")
_VIZ_TRACKER_AVAILABLE = RequirementCache("viztracer")
_LIGHTNING_CLOUD_LATEST = RequirementCache("lightning-cloud>=0.5.51")
_BOTO3_AVAILABLE = RequirementCache("boto3")

# DON'T CHANGE ORDER
_TORCH_DTYPES_MAPPING = {
    0: torch.float32,
    1: torch.float,
    2: torch.float64,
    3: torch.double,
    4: torch.complex64,
    5: torch.cfloat,
    6: torch.complex128,
    7: torch.cdouble,
    8: torch.float16,
    9: torch.half,
    10: torch.bfloat16,  # Not supported https://github.com/pytorch/pytorch/issues/110285
    11: torch.uint8,
    12: torch.int8,
    13: torch.int16,
    14: torch.short,
    15: torch.int32,
    16: torch.int,
    17: torch.int64,
    18: torch.long,
    19: torch.bool,
}
