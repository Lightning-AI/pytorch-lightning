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
import pytest
import torch

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core import memory
from tests.base import EvalModelTemplate
import pytorch_lightning as pl


PRETEND_N_OF_GPUS = 16


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_early_stop_dp(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping()],
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator='dp',
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
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

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile('min_max')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_dp_test(tmpdir):
    tutils.set_random_master_port()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    model = EvalModelTemplate()
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator='dp',
    )
    trainer.fit(model)
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path
    results = trainer.test()
    assert 'test_acc' in results[0]

    old_weights = model.c_d1.weight.clone().detach().cpu()

    results = trainer.test(model)
    assert 'test_acc' in results[0]

    # make sure weights didn't change
    new_weights = model.c_d1.weight.clone().detach().cpu()

    assert torch.all(torch.eq(old_weights, new_weights))
