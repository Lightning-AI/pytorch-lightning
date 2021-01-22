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
# limitations under the License

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils
from tests.base.boring_model import BoringModel
from tests.base.develop_utils import pl_multi_process_test


@pytest.mark.skipif(not XLADeviceUtils.tpu_device_exists(), reason="test requires TPU machine")
@pl_multi_process_test
def test_resume_training_on_cpu(tmpdir):
    """ Checks if training can be resumed from a saved checkpoint on CPU"""

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=8,)
    trainer.fit(model)

    model_path = trainer.checkpoint_callback.best_model_path

    # Verify saved Tensors are on CPU
    ckpt = torch.load(model_path)
    weight_tensor = list(ckpt["state_dict"].values())[0]
    assert weight_tensor.device == torch.device("cpu")

    # Verify that training is resumed on CPU
    trainer = Trainer(
        resume_from_checkpoint=model_path,
        checkpoint_callback=True,
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skipif(not XLADeviceUtils.tpu_device_exists(), reason="test requires TPU machine")
@pl_multi_process_test
def test_if_test_works_after_train(tmpdir):
    """ Ensure that .test() works after .fit() """

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=8, default_root_dir=tmpdir)
    trainer.fit(model)

    assert trainer.test() == 1
