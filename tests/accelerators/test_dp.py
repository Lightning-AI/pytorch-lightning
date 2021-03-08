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
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core import memory
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel

PRETEND_N_OF_GPUS = 16


class CustomClassificationModelDP(ClassificationModel):

    def _step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return {'logits': logits, 'y': y}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, batch_idx)
        loss = F.cross_entropy(out['logits'], out['y'])
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step_end(self, outputs):
        self.log('val_acc', self.valid_acc(outputs['logits'], outputs['y']))

    def test_step_end(self, outputs):
        self.log('test_acc', self.test_acc(outputs['logits'], outputs['y']))


@RunIf(min_gpus=2)
def test_multi_gpu_early_stop_dp(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor='val_acc')],
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator='dp',
    )

    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_gpus=2)
def test_multi_gpu_model_dp(tmpdir):
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator='dp',
        progress_bar_refresh_rate=0,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile('min_max')


@RunIf(min_gpus=2)
def test_dp_test(tmpdir):
    tutils.set_random_master_port()

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator='dp',
    )
    trainer.fit(model, datamodule=dm)
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path
    results = trainer.test(datamodule=dm)
    assert 'test_acc' in results[0]

    old_weights = model.layer_0.weight.clone().detach().cpu()

    results = trainer.test(model, datamodule=dm)
    assert 'test_acc' in results[0]

    # make sure weights didn't change
    new_weights = model.layer_0.weight.clone().detach().cpu()

    assert torch.all(torch.eq(old_weights, new_weights))


@RunIf(min_gpus=2)
def test_dp_training_step_dict(tmpdir):
    """
    This test verify dp properly reduce dictionaries
    """

    model = BoringModel()
    model.training_step_end = None
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        gpus=2,
        accelerator='dp',
    )
    trainer.fit(model)
