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

import time
from typing import Type

import pytest
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.plugins import DDPSpawnShardedPlugin
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class SeedTrainLoaderModel(BoringModel):
    """
    Overrides training loader to ensure we enforce the same seed for all DDP processes.
    """

    def train_dataloader(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


class SeedTrainLoaderManualModel(SeedTrainLoaderModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        # manual
        # access your optimizers with use_pl_optimizer=False. Default is True
        (opt_a, opt_b) = self.optimizers(use_pl_optimizer=True)
        loss_1 = self.step(batch)

        self.manual_backward(loss_1)
        opt_a.step()

        # fake discriminator
        loss_2 = self.step(batch[0])

        # ensure we forward the correct params to the optimizer
        # without retain_graph we can't do multiple backward passes
        self.manual_backward(loss_2)
        # todo: understand why synchronization breaks there.
        # self.manual_backward(loss_2, retain_graph=True)
        opt_b.step()

        assert self.layer.weight.grad is None or torch.all(self.layer.weight.grad == 0)

    def training_epoch_end(self, outputs) -> None:
        # outputs should be an array with an entry per optimizer
        assert len(outputs) == 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2

    @property
    def automatic_optimization(self) -> bool:
        return False


class SeedTrainLoaderMultipleOptimizersModel(SeedTrainLoaderModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        # outputs should be an array with an entry per optimizer
        assert len(outputs) == 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2


def record_ddp_fit_model_stats(trainer, model, use_cuda):
    """
    Helper to calculate wall clock time for fit + max allocated memory.

    Args:
        trainer: The trainer object.
        model: The model to fit.
        use_cuda: Whether to sync CUDA kernels.

    Returns:
        Max Memory if using GPUs, and total wall clock time.
    """
    max_memory = None

    time_start = time.perf_counter()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    trainer.fit(model)

    if use_cuda:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2 ** 20

    total_time = time.perf_counter() - time_start

    return max_memory, total_time


def plugin_parity_test(
    model_cls: Type[SeedTrainLoaderModel],
    seed: int = 42,
    gpus: int = 0,
    precision: int = 32,
    max_percent_speed_diff: float = 0.1,
):
    """
    Ensures that the trained model is identical to the standard DDP implementation.
    Also checks for speed/memory regressions, we should expect always less memory but performance to fluctuate.

    Args:
        model_cls: Model class to use for test.
        seed: Seed for generators. Note that this does not handle the seed for data-loading on multi-process.
        gpus: Number of GPUS to enable.
        precision: Whether to use AMP or normal FP32 training.
        max_percent_speed_diff: The maximum speed difference compared to normal DDP training.
        This is more a safety net for variability in CI which can vary in speed, not for benchmarking.

    """

    # Train normal DDP
    seed_everything(seed)
    ddp_model = model_cls()
    use_cuda = gpus > 0

    trainer = Trainer(fast_dev_run=True, max_epochs=1, gpus=gpus, precision=precision, accelerator="ddp_spawn")

    max_memory_ddp, ddp_time = record_ddp_fit_model_stats(trainer=trainer, model=ddp_model, use_cuda=use_cuda)

    # Reset and train Custom DDP
    seed_everything(seed)
    custom_plugin_model = model_cls()

    trainer = Trainer(fast_dev_run=True, max_epochs=1, gpus=gpus, precision=precision, accelerator="ddp_sharded_spawn")
    assert isinstance(trainer.training_type_plugin, DDPSpawnShardedPlugin)

    max_memory_custom, custom_model_time = record_ddp_fit_model_stats(
        trainer=trainer, model=custom_plugin_model, use_cuda=use_cuda
    )

    # Assert model parameters are identical after fit
    for ddp_param, custom_param in zip(ddp_model.parameters(), custom_plugin_model.parameters()):
        assert torch.equal(ddp_param, custom_param), "Model parameters are different between DDP and Custom plugin"

    # Assert speed parity by ensuring percentage difference between custom/ddp is below threshold
    percent_diff = (custom_model_time - ddp_time) / custom_model_time

    assert (
        percent_diff <= max_percent_speed_diff
    ), f"Custom DDP plugin was too slow compared to DDP, Custom Plugin Time: {custom_model_time}, DDP Time: {ddp_time}"

    if use_cuda:
        # Assert CUDA memory parity
        assert max_memory_custom <= max_memory_ddp, (
            "Custom plugin used too much memory compared to DDP, "
            f"Custom Mem: {max_memory_custom}, DDP Mem: {max_memory_ddp}"
        )


@RunIf(skip_windows=True, fairscale=True)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(gpus=1, model_cls=SeedTrainLoaderModel), marks=RunIf(min_gpus=1)),
        pytest.param(
            dict(gpus=1, precision=16, model_cls=SeedTrainLoaderModel), marks=RunIf(min_gpus=1, amp_native=True)
        ),
        pytest.param(dict(gpus=2, model_cls=SeedTrainLoaderModel), marks=RunIf(min_gpus=2)),
        pytest.param(
            dict(gpus=2, precision=16, model_cls=SeedTrainLoaderModel), marks=RunIf(min_gpus=2, amp_native=True)
        ),
        pytest.param(
            dict(gpus=2, model_cls=SeedTrainLoaderMultipleOptimizersModel),
            marks=[
                RunIf(min_gpus=2),
                pytest.mark.skip(reason="TODO: Current issue with multiple optimizers and FairScale."),
            ],
        ),
        pytest.param(
            dict(gpus=2, model_cls=SeedTrainLoaderManualModel),
            marks=[
                RunIf(min_gpus=2),
                pytest.mark.skip(reason="TODO: Current issue with multiple optimizers and FairScale."),
            ],
        ),
    ],
)
def test_ddp_spawn_sharded_plugin(kwargs):
    if kwargs["gpus"] > 1:
        # TODO: decrease speed diff since only 2 GPUs sharding 2 optimizers
        kwargs["max_percent_speed_diff"] = 0.25
    plugin_parity_test(**kwargs)
