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
from abc import ABC

import pytorch_lightning as pl
from lightning_lite.accelerators.accelerator import Accelerator as _Accelerator


class Accelerator(_Accelerator, ABC):
    """The Accelerator base class for Lightning PyTorch. An Accelerator is meant to deal with one type of Hardware.

    Currently, there are accelerators for:

    - CPU
    - GPU
    - TPU
    - IPU
    - HPU
    """

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """
