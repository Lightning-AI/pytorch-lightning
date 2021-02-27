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
"""Test deprecated functionality which will be removed in v1.5.0"""

from unittest import mock

import pytest

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
from tests.helpers import BoringModel
from tests.helpers.utils import no_warning_call


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_v1_5_0_wandb_unused_sync_step(tmpdir):
    with pytest.deprecated_call(match=r"v1.2.1 and will be removed in v1.5"):
        WandbLogger(sync_step=True)


def test_v1_5_0_old_callback_on_save_checkpoint(tmpdir):
    class OldSignature(Callback):
        def on_save_checkpoint(self, trainer, pl_module):  # noqa
            ...

    model = BoringModel()
    trainer_kwargs = {
        "default_root_dir": tmpdir,
        "checkpoint_callback": False,
        "max_epochs": 1,
    }
    filepath = tmpdir / "test.ckpt"

    trainer = Trainer(**trainer_kwargs, callbacks=[OldSignature()])
    trainer.fit(model)

    with pytest.deprecated_call(match="old signature will be removed in v1.5"):
        trainer.save_checkpoint(filepath)

    class NewSignature(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            ...

    class ValidSignature1(Callback):
        def on_save_checkpoint(self, trainer, *args):
            ...

    class ValidSignature2(Callback):
        def on_save_checkpoint(self, *args):
            ...

    trainer.callbacks = [NewSignature(), ValidSignature1(), ValidSignature2()]
    with no_warning_call(DeprecationWarning):
        trainer.save_checkpoint(filepath)
