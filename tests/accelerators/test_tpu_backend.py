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
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerState
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


class WeightSharingModule(BoringModel):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


@RunIf(tpu=True)
@pl_multi_process_test
def test_resume_training_on_cpu(tmpdir):
    """ Checks if training can be resumed from a saved checkpoint on CPU"""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(
        checkpoint_callback=True,
        max_epochs=1,
        tpu_cores=8,
    )
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
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


@RunIf(tpu=True)
@pl_multi_process_test
def test_if_test_works_after_train(tmpdir):
    """ Ensure that .test() works after .fit() """

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, tpu_cores=8, default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert len(trainer.test(model)) == 1


@RunIf(tpu=True)
@pl_multi_process_test
def test_weight_tying_warning(tmpdir, capsys=None):
    """
    Ensure a warning is thrown if model parameter lengths do not match
    post moving to device.
    """

    model = WeightSharingModule()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=1)

    with pytest.warns(UserWarning, match=r'The model layers do not match after moving to the target device.'):
        result = trainer.fit(model)
        assert result


# @RunIf(tpu=True)
# @pl_multi_process_test
# def test_if_weights_tied(tmpdir, capsys=None):
#     """
#     Test if weights are properly tied on `on_post_move_to_device`.
#     Ensure no warning for parameter mismatch is thrown.
#     """

#     # TODO (kaushikb11): Add `paramter_validation` specific to
#     # TPU Accelerators
#     class Model(WeightSharingModule):

#         def on_post_move_to_device(self):
#             self.layer_3.weight = self.layer_1.weight

#     model = Model()
#     trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=1)

#     with pytest.warns(UserWarning) as warnings:
#         result = trainer.fit(model)
#         assert result

#     assert not list(filter(lambda x: 'The model layers do not match' in str(x), warnings.list))
#     assert len(trainer.test(model)) == 1
