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
from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel


def test_on_val_epoch_end_outputs(tmpdir):

    class TestModel(BoringModel):

        def on_validation_epoch_end(self, outputs):
            if trainer.running_sanity_check:
                assert len(outputs[0]) == trainer.num_sanity_val_batches[0]
            else:
                assert len(outputs[0]) == trainer.num_val_batches[0]

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        weights_summary=None,
    )

    trainer.fit(model)


def test_on_test_epoch_end_outputs(tmpdir):

    class TestModel(BoringModel):

        def on_test_epoch_end(self, outputs):
            assert len(outputs[0]) == trainer.num_test_batches[0]

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=2,
        weights_summary=None,
    )

    trainer.test(model)
