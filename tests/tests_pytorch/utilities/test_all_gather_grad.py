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

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.core.test_results import spawn_launch
from tests_pytorch.helpers.runif import RunIf


def all_gather_ddp_spawn_fn(strategy):
    rank = strategy.local_rank
    world_size = strategy.num_processes

    tensor1 = torch.ones(8, requires_grad=True)
    tensor2 = torch.ones((8, 16, 32), requires_grad=True)

    tensor1_gathered = strategy.all_gather(tensor1, sync_grads=True)
    tensor2_gathered = strategy.all_gather(tensor2, sync_grads=True)

    tensor1_gathered = tensor1_gathered * rank
    tensor2_gathered = tensor2_gathered * rank

    tensor1_gathered.sum().backward()
    tensor2_gathered.sum().backward()

    grad1 = torch.zeros_like(tensor1.grad).fill_(torch.arange(world_size).sum().float())
    grad2 = torch.zeros_like(tensor2.grad).fill_(torch.arange(world_size).sum().float())

    assert torch.allclose(grad1, tensor1.grad)
    assert torch.allclose(grad2, tensor2.grad)


@RunIf(skip_windows=True)
@pytest.mark.flaky(reruns=3)
def test_all_gather_ddp_spawn():
    spawn_launch(all_gather_ddp_spawn_fn, [torch.device("cpu")] * 3)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_all_gather_collection(tmpdir):
    class TestModel(BoringModel):
        on_train_epoch_end_called = False

        def on_train_epoch_end(self):
            losses = torch.rand(2, 2).t()
            gathered_loss = self.all_gather(
                {
                    "losses_tensor_int": losses.int(),
                    "losses_tensor_float": losses,
                    "losses_tensor_list": [losses, losses],
                    "losses_np_ndarray": np.array([1, 2, 3]),
                    "losses_bool": [True, False],
                    "losses_float": [0.0, 1.0, 2.0],
                    "losses_int": [0, 1, 2],
                }
            )
            assert gathered_loss["losses_tensor_int"][0].dtype == torch.int32
            assert gathered_loss["losses_tensor_float"][0].dtype == torch.float
            assert gathered_loss["losses_np_ndarray"][0].dtype == torch.int64
            # torch.bool can't be all_gathered
            assert gathered_loss["losses_bool"][0].dtype == torch.uint8
            assert gathered_loss["losses_float"][0].dtype == torch.float
            assert gathered_loss["losses_int"][0].dtype == torch.int

            losses_numel = losses.numel()
            assert gathered_loss["losses_tensor_int"].numel() == 2 * losses_numel
            assert gathered_loss["losses_tensor_float"].numel() == 2 * losses_numel
            assert torch.stack(gathered_loss["losses_tensor_list"]).shape == (2, 2, 2, 2)
            assert gathered_loss["losses_np_ndarray"].numel() == 2 * 3
            assert torch.stack(gathered_loss["losses_bool"]).shape == (2, 2)
            assert torch.stack(gathered_loss["losses_float"]).shape == (3, 2)
            assert torch.stack(gathered_loss["losses_int"]).shape == (3, 2)
            self.on_train_epoch_end_called = True

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=8,
        limit_val_batches=0,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    assert model.on_train_epoch_end_called


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_all_gather_sync_grads(tmpdir):
    class TestModel(BoringModel):
        training_step_called = False

        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            tensor = torch.rand(2, 2, requires_grad=True, device=self.device)
            gathered_tensor = self.all_gather(tensor, sync_grads=True)
            assert gathered_tensor.shape == torch.Size([2, 2, 2])
            return gathered_tensor.sum()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    assert model.training_step_called
