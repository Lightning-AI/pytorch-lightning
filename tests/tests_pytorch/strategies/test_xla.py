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
import os
from unittest import mock

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import XLAAccelerator
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import XLAStrategy
from tests_pytorch.helpers.runif import RunIf


class BoringModelTPU(BoringModel):
    def on_train_start(self) -> None:
        from torch_xla.experimental import pjrt

        index = 0 if pjrt.using_pjrt() else 1
        # assert strategy attributes for device setting
        assert self.device == torch.device("xla", index=index)
        assert os.environ.get("PT_XLA_DEBUG") == "1"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_xla_strategy_debug_state():
    """Tests if device/debug flag is set correctly when training and after teardown for XLAStrategy."""
    model = BoringModelTPU()
    from torch_xla.experimental import pjrt

    trainer_kwargs = {}
    if not pjrt.using_pjrt():
        # only XRT supports XLA with a single process
        trainer_kwargs["devices"] = 1
    trainer = Trainer(fast_dev_run=True, strategy=XLAStrategy(debug=True), **trainer_kwargs)
    assert isinstance(trainer.accelerator, XLAAccelerator)
    assert isinstance(trainer.strategy, XLAStrategy)
    trainer.fit(model)
    assert "PT_XLA_DEBUG" not in os.environ
