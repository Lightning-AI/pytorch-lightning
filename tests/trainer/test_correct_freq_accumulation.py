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
"""
Tests to ensure that the training loop works with a dict
"""
import os
from unittest import mock

from pytorch_lightning import Trainer
from tests.base.model_template import EvalModelTemplate


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_training_step_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """

    model = EvalModelTemplate()
    model.validation_step = None
    model.test_step = None
    model.training_step = model.training_step_result_obj_dp
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_obj_dp
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # epoch 0
    assert trainer.dev_debugger.logged_metrics[0]['global_step'] == 0
    assert trainer.dev_debugger.logged_metrics[1]['global_step'] == 1
    assert trainer.dev_debugger.logged_metrics[2]['global_step'] == 1

    # epoch 1
    assert trainer.dev_debugger.logged_metrics[3]['global_step'] == 2
    assert trainer.dev_debugger.logged_metrics[4]['global_step'] == 3
    assert trainer.dev_debugger.logged_metrics[5]['global_step'] == 3
