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
import torch
from torch import optim

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, BaseProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.trainer.callback_hook import warning_cache as callback_warning_cache
from tests.deprecated_api import no_deprecated_call
from tests.helpers import BoringModel
from tests.helpers.utils import no_warning_call


def test_v1_5_0_model_checkpoint_save_checkpoint():
    model_ckpt = ModelCheckpoint()
    model_ckpt.save_function = lambda *_, **__: None
    with pytest.deprecated_call(match="ModelCheckpoint.save_checkpoint` signature has changed"):
        model_ckpt.save_checkpoint(Trainer(), object())


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


def test_v1_5_0_legacy_profiler_argument():
    with pytest.deprecated_call(match="renamed to `record_functions` in v1.3"):
        PyTorchProfiler(profiled_functions=[])


def test_v1_5_0_running_sanity_check():
    trainer = Trainer()
    with pytest.deprecated_call(match='has been renamed to `Trainer.sanity_checking`'):
        assert not trainer.running_sanity_check


def test_old_training_step_signature_with_opt_idx_manual_opt(tmpdir):

    class OldSignatureModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            assert optimizer_idx is not None
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            return [optim.SGD(self.parameters(), lr=1e-2), optim.SGD(self.parameters(), lr=1e-2)]

    model = OldSignatureModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)

    with pytest.deprecated_call(match="`training_step` .* `optimizer_idx` .* manual .* will be removed in v1.5"):
        trainer.fit(model)


def test_v1_5_0_model_checkpoint_period(tmpdir):
    with no_warning_call(DeprecationWarning):
        ModelCheckpoint(dirpath=tmpdir)
    with pytest.deprecated_call(match="is deprecated in v1.3 and will be removed in v1.5"):
        ModelCheckpoint(dirpath=tmpdir, period=1)


def test_v1_5_0_old_on_validation_epoch_end(tmpdir):
    callback_warning_cache.clear()

    class OldSignature(Callback):

        def on_validation_epoch_end(self, trainer, pl_module):  # noqa
            ...

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=OldSignature())

    with pytest.deprecated_call(match="old signature will be removed in v1.5"):
        trainer.fit(model)

    class OldSignatureModel(BoringModel):

        def on_validation_epoch_end(self):  # noqa
            ...

    model = OldSignatureModel()

    with pytest.deprecated_call(match="old signature will be removed in v1.5"):
        trainer.fit(model)

    callback_warning_cache.clear()

    class NewSignature(Callback):

        def on_validation_epoch_end(self, trainer, pl_module, outputs):
            ...

    trainer.callbacks = [NewSignature()]
    with no_deprecated_call(match="`Callback.on_validation_epoch_end` signature has changed in v1.3."):
        trainer.fit(model)

    class NewSignatureModel(BoringModel):

        def on_validation_epoch_end(self, outputs):
            ...

    model = NewSignatureModel()
    with no_deprecated_call(match="`ModelHooks.on_validation_epoch_end` signature has changed in v1.3."):
        trainer.fit(model)


def test_v1_5_0_old_on_test_epoch_end(tmpdir):
    callback_warning_cache.clear()

    class OldSignature(Callback):

        def on_test_epoch_end(self, trainer, pl_module):  # noqa
            ...

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=OldSignature())

    with pytest.deprecated_call(match="old signature will be removed in v1.5"):
        trainer.test(model)

    class OldSignatureModel(BoringModel):

        def on_test_epoch_end(self):  # noqa
            ...

    model = OldSignatureModel()

    with pytest.deprecated_call(match="old signature will be removed in v1.5"):
        trainer.test(model)

    callback_warning_cache.clear()

    class NewSignature(Callback):

        def on_test_epoch_end(self, trainer, pl_module, outputs):
            ...

    trainer.callbacks = [NewSignature()]
    with no_deprecated_call(match="`Callback.on_test_epoch_end` signature has changed in v1.3."):
        trainer.test(model)

    class NewSignatureModel(BoringModel):

        def on_test_epoch_end(self, outputs):
            ...

    model = NewSignatureModel()
    with no_deprecated_call(match="`ModelHooks.on_test_epoch_end` signature has changed in v1.3."):
        trainer.test(model)


@pytest.mark.parametrize("cls", (BaseProfiler, SimpleProfiler, AdvancedProfiler, PyTorchProfiler))
def test_v1_5_0_profiler_output_filename(tmpdir, cls):
    filepath = str(tmpdir / "test.txt")
    with pytest.deprecated_call(match="`output_filename` parameter has been removed"):
        profiler = cls(output_filename=filepath)
    assert profiler.dirpath == tmpdir
    assert profiler.filename == "test"


def test_v1_5_0_trainer_training_trick_mixin(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    with pytest.deprecated_call(match="is deprecated in v1.3 and will be removed in v1.5"):
        trainer.print_nan_gradients()

    dummy_loss = torch.tensor(1.0)
    with pytest.deprecated_call(match="is deprecated in v1.3 and will be removed in v1.5"):
        trainer.detect_nan_tensors(dummy_loss)


def test_v1_5_0_auto_move_data():
    with pytest.deprecated_call(match="deprecated in v1.3 and will be removed in v1.5.*was applied to `bar`"):

        class Foo:

            @auto_move_data
            def bar(self):
                pass
