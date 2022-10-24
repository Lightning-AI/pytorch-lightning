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
from copy import deepcopy
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests_lite.helpers.models import BoringLite, RandomDataset, RandomIterableDataset
from tests_lite.helpers.runif import RunIf
from tests_lite.test_lite import BoringModel
from torch.utils.data import DataLoader

from lightning_lite import LightningLite
from lightning_lite.plugins import DeepSpeedPrecision
from lightning_lite.strategies import DeepSpeedStrategy


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multiple_models():
    class Lite(LightningLite):
        def run(self):
            model = BoringModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model, optimizer = self.setup(model, optimizer)

            for i in range(2):
                optimizer.zero_grad()
                x = model(torch.randn(1, 32).to(self.device))
                loss = x.sum()
                if i == 0:
                    # the weights are not initialized with stage 3 until backward is run once
                    assert all(w.nelement() == 0 for w in model.state_dict().values())
                self.backward(loss, model=model)
                if i == 0:
                    # save for later to check that the weights were updated
                    state_dict = deepcopy(model.state_dict())
                optimizer.step()

            # check that the model trained, the weights from step 1 do not match the weights from step 2
            for mw_b, mw_a in zip(state_dict.values(), model.state_dict().values()):
                assert not torch.allclose(mw_b, mw_a)

            self.seed_everything(42)
            model_1 = BoringModel()
            optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.0001)

            self.seed_everything(42)
            model_2 = BoringModel()
            optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.0001)

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.allclose(mw_1, mw_2)

            model_1, optimizer_1 = self.setup(model_1, optimizer_1)
            model_2, optimizer_2 = self.setup(model_2, optimizer_2)

            # train model_1 first
            self.seed_everything(42)
            data_list = []
            for _ in range(2):
                optimizer_1.zero_grad()
                data = torch.randn(1, 32).to(self.device)
                data_list.append(data)
                x = model_1(data)
                loss = x.sum()
                self.backward(loss, model=model_1)
                optimizer_1.step()

            # the weights do not match
            assert all(w.nelement() > 1 for w in model_1.state_dict().values())
            assert all(w.nelement() == 0 for w in model_2.state_dict().values())

            # now train model_2 with the same data
            for data in data_list:
                optimizer_2.zero_grad()
                x = model_2(data)
                loss = x.sum()
                self.backward(loss, model=model_2)
                optimizer_2.step()

            # the weights should match
            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.allclose(mw_1, mw_2)

            # Verify collectives works as expected
            ranks = self.all_gather(torch.tensor([self.local_rank]).to(self.device))
            assert torch.allclose(ranks.cpu(), torch.tensor([[0], [1]]))
            assert self.broadcast(True)
            assert self.is_global_zero == (self.local_rank == 0)

    Lite(strategy=DeepSpeedStrategy(stage=3, logging_batch_size_per_gpu=1), devices=2, accelerator="gpu").run()


@RunIf(min_cuda_gpus=1, deepspeed=True)
@pytest.mark.parametrize(
    ["dataset_cls", "logging_batch_size_per_gpu", "expected_batch_size"],
    [
        (RandomDataset, None, 1),
        (RandomDataset, 10, 10),
        (RandomIterableDataset, None, 1),
        (RandomIterableDataset, 10, 10),
    ],
)
def test_deepspeed_auto_batch_size_config_select(dataset_cls, logging_batch_size_per_gpu, expected_batch_size):
    """Test to ensure that the batch size is correctly set as expected for deepspeed logging purposes."""

    class Lite(LightningLite):
        def run(self):
            assert isinstance(self._strategy, DeepSpeedStrategy)
            _ = self.setup_dataloaders(DataLoader(dataset_cls(32, 64)))
            config = self._strategy.config
            assert config["train_micro_batch_size_per_gpu"] == expected_batch_size

    lite = Lite(
        accelerator="cuda",
        devices=1,
        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=logging_batch_size_per_gpu, zero_optimization=False),
    )
    lite.run()


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_configure_optimizers():
    """Test that the deepspeed strategy with default initialization wraps the optimizer correctly."""

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    class Lite(LightningLite):
        def run(self):
            model = nn.Linear(3, 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            model, optimizer = self.setup(model, optimizer)
            assert isinstance(optimizer.optimizer, DeepSpeedZeroOptimizer)
            assert isinstance(optimizer.optimizer.optimizer, torch.optim.SGD)

    lite = Lite(
        strategy=DeepSpeedStrategy(),
        accelerator="cuda",
        devices=1,
        precision=16,
    )
    lite.run()


@RunIf(min_cuda_gpus=1, deepspeed=True)
def test_deepspeed_custom_precision_params():
    """Test that if the FP16 parameters are set via the DeepSpeedStrategy, the deepspeed config contains these
    changes."""

    class Lite(LightningLite):
        def run(self):
            assert self._strategy._config_initialized
            assert self._strategy.config["fp16"]["loss_scale"] == 10
            assert self._strategy.config["fp16"]["initial_scale_power"] == 11
            assert self._strategy.config["fp16"]["loss_scale_window"] == 12
            assert self._strategy.config["fp16"]["hysteresis"] == 13
            assert self._strategy.config["fp16"]["min_loss_scale"] == 14

    strategy = DeepSpeedStrategy(
        loss_scale=10, initial_scale_power=11, loss_scale_window=12, hysteresis=13, min_loss_scale=14
    )
    lite = Lite(
        strategy=strategy,
        precision=16,
        accelerator="cuda",
        devices=1,
    )
    lite.run()


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params_forwarded():
    """Test that the activation checkpointing parameters get passed to `deepspeed.checkpointing.configure`
    correctly."""
    import deepspeed

    class Lite(LightningLite):
        def run(self):
            model = nn.Linear(3, 3)
            optimizer = torch.optim.Adam(model.parameters())

            with mock.patch("deepspeed.checkpointing.configure", wraps=deepspeed.checkpointing.configure) as configure:
                self.setup(model, optimizer)

            configure.assert_called_with(
                mpu_=None,
                partition_activations=True,
                contiguous_checkpointing=True,
                checkpoint_in_cpu=True,
                profile=None,
            )

    strategy = DeepSpeedStrategy(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True,
    )
    lite = Lite(
        strategy=strategy,
        precision=16,
        accelerator="cuda",
        devices=1,
    )
    lite.run()


class ModelParallelClassification(BoringLite):

    num_blocks = 5

    def get_model(self):
        return nn.Sequential(*(self._make_block() for _ in range(self.num_blocks)), nn.Linear(32, 3))

    def step(self, model, batch):
        x = batch
        y = torch.ones(batch.size(0), device=batch.device, dtype=torch.long)
        x = model(x)
        # Ensure output is in float32 for softmax operation
        x = x.float()
        logits = F.softmax(x, dim=1)
        loss = F.cross_entropy(logits, y)
        return loss

    def _make_block(self):
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3(tmpdir):
    """Test to ensure ZeRO Stage 3 works with a parallel model."""
    lite = ModelParallelClassification(
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="cuda",
        devices=2,
        precision=16,
    )
    lite.run()


@RunIf(deepspeed=True)
@mock.patch("deepspeed.init_distributed", autospec=True)
@pytest.mark.parametrize("platform", ["Linux", "Windows"])
def test_deepspeed_env_variables_on_platforms(deepspeed_dist_mock, tmpdir, platform):
    """Test to ensure that we set up distributed communication correctly.

    When using Windows, ranks environment variables should not be set, and DeepSpeed should handle this.
    """
    lite = BoringLite(strategy=DeepSpeedStrategy(stage=3))
    strategy = lite._strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    with mock.patch("platform.system", return_value=platform) as platform_mock:
        strategy._init_deepspeed_distributed()
    deepspeed_dist_mock.assert_called()
    platform_mock.assert_called()
    if platform == "Windows":
        # assert no env variables have been set within the DeepSpeedStrategy
        assert all(k not in os.environ for k in ("MASTER_PORT", "MASTER_ADDR", "RANK", "WORLD_SIZE", "LOCAL_RANK"))
    else:
        assert os.environ["MASTER_ADDR"] == str(strategy.cluster_environment.main_address)
        assert os.environ["MASTER_PORT"] == str(strategy.cluster_environment.main_port)
        assert os.environ["RANK"] == str(strategy.global_rank)
        assert os.environ["WORLD_SIZE"] == str(strategy.world_size)
        assert os.environ["LOCAL_RANK"] == str(strategy.local_rank)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_specific_gpu_device_index(tmpdir):
    """Test that the DeepSpeed strategy can run on specific device indices."""

    class Lite(BoringLite):
        def step(self, model, batch):
            assert self.device.type == "cuda"
            assert self.device.index == 1
            assert batch.device.index == 1
            assert model.device.index == 1
            return super().step(model, batch)

    lite = Lite(accelerator="cuda", devices=[1], strategy="deepspeed")
    lite.run()


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
def test_deepspeed_with_bfloat16_precision(tmpdir):
    """Test that the DeepSpeed strategy works with bfloat16 precision."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 2)

        def forward(self, x):
            assert x.dtype == torch.bfloat16
            return self.layer(x)

    class Lite(BoringLite):
        def get_model(self):
            return Model()

        def step(self, model, batch):
            assert self._strategy.config["bf16"]["enabled"]
            assert batch.dtype == torch.float32
            assert model.layer.weight.dtype == torch.bfloat16
            return super().step(model, batch)

    lite = Lite(accelerator="cuda", devices=2, strategy="deepspeed_stage_3", precision="bf16")
    assert isinstance(lite._strategy.precision, DeepSpeedPrecision)
    assert lite._strategy.precision.precision == "bf16"
    assert lite._strategy.config["zero_optimization"]["stage"] == 3
    lite.run()
