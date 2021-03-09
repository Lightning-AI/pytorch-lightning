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
from unittest import mock

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel


@mock.patch('pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval')
@mock.patch('pytorch_lightning.core.hooks.ModelHooks.on_validation_model_train')
@mock.patch('pytorch_lightning.core.hooks.ModelHooks.on_test_model_eval')
@mock.patch('pytorch_lightning.core.hooks.ModelHooks.on_test_model_train')
def test_eval_train_calls(test_train_mock, test_eval_mock, val_train_mock, val_eval_mock, tmpdir):
    """
    Tests that only training_step can be used
    """
    model = BoringModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)
    trainer.test()

    # sanity + 2 epochs
    assert val_eval_mock.call_count == 3
    assert val_train_mock.call_count == 3

    # test is called only once
    assert test_eval_mock.call_count == 1
    assert test_train_mock.call_count == 1
