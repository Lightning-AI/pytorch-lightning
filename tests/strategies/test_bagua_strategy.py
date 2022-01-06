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
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import BaguaStrategy
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 32)

    def test_epoch_end(self, outputs) -> None:
        mean_y = torch.stack([x["y"] for x in outputs]).mean()
        self.log("mean_y", mean_y)


class TestModel4QAdam(TestModel):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.05, warmup_steps=20)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@RunIf(skip_windows=True, bagua=True, min_gpus=1, standalone=True)
def test_bagua_default():
    model = TestModel()
    trainer = Trainer(max_epochs=1, strategy="bagua", gpus=1)
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] < 2


@pytest.mark.parametrize(
    "algorithm", ["gradient_allreduce", "bytegrad", "decentralized", "low_precision_decentralized"]
)
@RunIf(skip_windows=True, bagua=True, min_gpus=2, standalone=True)
def test_bagua_algorithm(algorithm):
    model = TestModel()
    bagua_strategy = BaguaStrategy(algorithm=algorithm)
    trainer = Trainer(
        max_epochs=1,
        strategy=bagua_strategy,
        gpus=2,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] < 2


@RunIf(skip_windows=True, bagua=True, min_gpus=2, standalone=True)
def test_bagua_async():
    model = TestModel()
    bagua_strategy = BaguaStrategy(algorithm="async", warmup_steps=10, sync_interval_ms=10)
    trainer = Trainer(
        max_epochs=1,
        strategy=bagua_strategy,
        gpus=2,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] < 2


@RunIf(skip_windows=True, bagua=True, min_gpus=2, standalone=True)
def test_qadam():
    model = TestModel4QAdam()
    bagua_strategy = BaguaStrategy(algorithm="qadam")
    trainer = Trainer(
        max_epochs=1,
        strategy=bagua_strategy,
        gpus=2,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] < 5


@RunIf(skip_windows=True, bagua=True, min_gpus=2, standalone=True)
def test_bagua_reduce():
    from pytorch_lightning.utilities.distributed import ReduceOp

    trainer = Trainer(strategy="bagua", gpus=2)
    trainer.strategy.setup_environment()

    trainer.strategy.barrier()

    # faster this way
    reduce_ops = ["mean", "AVG", "undefined", "sum", ReduceOp.SUM, ReduceOp.MAX, ReduceOp.PRODUCT, ReduceOp.MIN]
    tensor = torch.randn(10).cuda()

    for reduce_op in reduce_ops:
        if reduce_op == "undefined":
            with pytest.raises(ValueError, match="unrecognized `reduce_op`"):
                trainer.strategy.reduce(tensor, reduce_op=reduce_op)
        else:
            tensor = trainer.strategy.reduce(tensor, reduce_op=reduce_op)
