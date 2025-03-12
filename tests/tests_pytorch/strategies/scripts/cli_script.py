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
"""A trivial script that wraps a LightningCLI around the BoringModel and BoringDataModule."""

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel

if __name__ == "__main__":
    LightningCLI(
        BoringModel,
        BoringDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )
