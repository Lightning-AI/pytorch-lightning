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
import warnings
from unittest import mock

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel


class TestModel(BoringModel):

    def training_step(self, batch, batch_idx):
        acc = self.step(batch[0])
        return acc


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_no_depre_without_epoch_end(tmpdir):
    """
    Tests that only training_step can be used
    """

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    with warnings.catch_warnings(record=True) as w:
        trainer.fit(model)

        for msg in w:
            assert 'should not return anything ' not in str(msg)
