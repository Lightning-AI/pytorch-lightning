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
from distutils.version import LooseVersion
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from tests.base.boring_model import BoringModel

try:
    breakpoint()
    import fairscale

    from pytorch_lightning.plugins.fairscale_pipe_pluging import PipePlugin
    HAS_FAIRSCALE = True
except Exception:
    HAS_FAIRSCALE = False


#@pytest.mark.skipif(not HAS_FAIRSCALE, reason="FairScale should be installed for this test",)
def test_pipe_plugin(tmpdir):

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_connector.backend, NativeAMPPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=0,
        distributed_backend='ddp_cpu',
        callbacks=[CB()],
        plugins=[PipePlugin()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)
