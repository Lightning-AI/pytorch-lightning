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

from pytorch_lightning.plugins.environments.cluster_environment import LightningEnvironment


class SageMakerEnvironment(LightningEnvironment):
    """Environment for distributed training on SageMaker."""

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])
