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

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import BaguaStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class BoringModel4QAdam(BoringModel):
    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.05, warmup_steps=20)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@RunIf(min_gpus=1, bagua=True)
def test_bagua_default(tmpdir):
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy="bagua",
        accelerator="gpu",
        devices=1,
    )
    assert isinstance(trainer.strategy, BaguaStrategy)


@RunIf(min_gpus=2, standalone=True, bagua=True)
def test_async_algorithm(tmpdir):
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="async")
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)

    for param in model.parameters():
        assert torch.norm(param) < 3


@RunIf(min_gpus=1, bagua=True)
@pytest.mark.parametrize(
    "algorithm", ["gradient_allreduce", "bytegrad", "qadam", "decentralized", "low_precision_decentralized"]
)
def test_configuration(algorithm, tmpdir):
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm=algorithm)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=1,
    )
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer

    with mock.patch(
        "bagua.torch_api.data_parallel.bagua_distributed.BaguaDistributedDataParallel.__init__", return_value=None
    ), mock.patch("bagua.torch_api.communication.is_initialized", return_value=True):
        if algorithm == "qadam":
            with pytest.raises(MisconfigurationException, match="Bagua QAdam can only accept one QAdamOptimizer"):
                trainer.strategy.configure_ddp()
        else:
            trainer.strategy.configure_ddp()


@RunIf(min_gpus=1, bagua=True)
def test_qadam_configuration(tmpdir):
    model = BoringModel4QAdam()
    bagua_strategy = BaguaStrategy(algorithm="qadam")
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=1,
    )
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_optimizers(trainer)

    with mock.patch(
        "bagua.torch_api.data_parallel.bagua_distributed.BaguaDistributedDataParallel.__init__", return_value=None
    ), mock.patch("bagua.torch_api.communication.is_initialized", return_value=True):
        trainer.strategy.configure_ddp()


def test_bagua_not_available(monkeypatch):
    import pytorch_lightning.strategies.bagua as imports

    monkeypatch.setattr(imports, "_BAGUA_AVAILABLE", False)
    with mock.patch("torch.cuda.device_count", return_value=1):
        with pytest.raises(MisconfigurationException, match="you must have `Bagua` installed"):
            Trainer(strategy="bagua", accelerator="gpu", devices=1)
