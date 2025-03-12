# Copyright The Lightning AI team.
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

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


def _device_check_helper(batch_device, module_device):
    assert batch_device.type == module_device.type
    if batch_device.index is not None and module_device.index is not None:
        assert batch_device.index == module_device.index
    else:
        # devices with index None are the same as with index 0
        assert batch_device.index in (0, None)
        assert module_device.index in (0, None)


class BatchHookObserverCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_validation_batch_start(self, trainer, pl_module, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_test_batch_start(self, trainer, pl_module, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_predict_batch_start(self, trainer, pl_module, batch, *_):
        _device_check_helper(batch.device, pl_module.device)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, *_):
        _device_check_helper(batch.device, pl_module.device)


class BatchHookObserverModel(BoringModel):
    def on_train_batch_start(self, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_train_batch_end(self, outputs, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_validation_batch_start(self, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_validation_batch_end(self, outputs, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_test_batch_start(self, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_test_batch_end(self, outputs, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_predict_batch_start(self, batch, *_):
        _device_check_helper(batch.device, self.device)

    def on_predict_batch_end(self, outputs, batch, *_):
        _device_check_helper(batch.device, self.device)


@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_callback_batch_on_device(tmp_path, accelerator):
    """Test that the batch object sent to the on_*_batch_start/end hooks is on the right device."""
    batch_callback = BatchHookObserverCallback()

    model = BatchHookObserverModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        accelerator=accelerator,
        devices=1,
        callbacks=[batch_callback],
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
