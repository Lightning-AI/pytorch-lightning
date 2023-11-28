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

from typing import Union

from lightning.data.constants import _LIGHTNING_CLOUD_LATEST
from lightning.data.processing.strategy.distributed.distributed import MultiNodeDataProcessorStrategy
from lightning.data.processing.strategy.single import SingleNodeDataProcessorStrategy
from lightning.data.streaming.cache import Dir
from lightning.data.utilities.env import _has_server

if _LIGHTNING_CLOUD_LATEST:
    from lightning_cloud.resolver import _resolve_dir


def _select_data_processor_strategy(input_dir: Union[str, Dir], output_dir: Union[str, Dir]):
    input_dir = _resolve_dir(input_dir)
    output_dir = _resolve_dir(output_dir)

    if _has_server():
        return MultiNodeDataProcessorStrategy(input_dir, output_dir)
    return SingleNodeDataProcessorStrategy(input_dir, output_dir)
