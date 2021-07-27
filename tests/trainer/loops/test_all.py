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
from pytorch_lightning import Callback, Trainer
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class BatchHookObserverCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, *args):
        assert batch.device == pl_module.device

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args):
        assert batch.device == pl_module.device

    def on_validation_batch_start(self, trainer, pl_module, batch, *args):
        assert batch.device == pl_module.device

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args):
        assert batch.device == pl_module.device

    def on_test_batch_start(self, trainer, pl_module, batch, *args):
        assert batch.device == pl_module.device

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args):
        assert batch.device == pl_module.device

    def on_predict_batch_start(self, trainer, pl_module, batch, *args):
        assert batch.device == pl_module.device

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, *args):
        assert batch.device == pl_module.device


class BatchHookObserverModel(BoringModel):
    def on_train_batch_start(self, batch, *args):
        assert batch.device == self.device

    def on_train_batch_end(self, outputs, batch, *args):
        assert batch.device == self.device

    def on_validation_batch_start(self, batch, *args):
        assert batch.device == self.device

    def on_validation_batch_end(self, outputs, batch, *args):
        assert batch.device == self.device

    def on_test_batch_start(self, batch, *args):
        assert batch.device == self.device

    def on_test_batch_end(self, outputs, batch, *args):
        assert batch.device == self.device

    def on_predict_batch_start(self, batch, *args):
        assert batch.device == self.device

    def on_predict_batch_end(self, outputs, batch, *args):
        assert batch.device == self.device


@RunIf(min_gpus=1)
def test_callback_batch_on_device(tmpdir):
    """Test that the batch object sent to the on_*_batch_start/end hooks is on the right device."""

    batch_callback = BatchHookObserverCallback()

    model = BatchHookObserverModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        gpus=1,
        callbacks=[batch_callback],
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
