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

r"""
Quantization
^^^^^^^^^^^^

"""
import torch

from pytorch_lightning import Callback


class StaticQuantization(Callback):

    def __init__(
        self,
    ) -> None:
        pass

    def on_fit_end(self, trainer, pl_module):
        # insert observers
        torch.quantization.prepare(pl_module, inplace=True)
        # Calibrate the model and collect statistics

        # convert to quantized version
        torch.quantization.convert(pl_module, inplace=True)
