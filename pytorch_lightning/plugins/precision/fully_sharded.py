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
from pytorch_lightning.plugins.precision.mixin import FullyShardedMixin
from pytorch_lightning.plugins.precision.sharded import ShardedBf16PrecisionPlugin, ShardedNativeMixedPrecisionPlugin


class FullyShardedNativeMixedPrecisionPlugin(ShardedNativeMixedPrecisionPlugin, FullyShardedMixin):
    """Fully Sharded support with native AMP."""


class FullyShardedBf16PrecisionPlugin(ShardedBf16PrecisionPlugin, FullyShardedMixin):
    """Fully Sharded support with bfloat16."""
