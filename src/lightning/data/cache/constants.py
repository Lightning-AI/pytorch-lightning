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

from lightning_utilities.core.imports import RequirementCache

_INDEX_FILENAME = "index.json"
_DEFAULT_CHUNK_BYTES = 1 << 26  # 64M B

# This is required for full pytree serialization / deserialization support
_TORCH_2_1_0_AVAILABLE = RequirementCache("torch>=2.1.0")
_VIZ_TRACKER_AVAILABLE = RequirementCache("viztracer")
