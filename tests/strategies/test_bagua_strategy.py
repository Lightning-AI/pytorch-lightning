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


class BoringModel4QAdam(BoringModel):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.1, warmup_steps=10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@RunIf(skip_windows=True, bagua=True, min_gpus=1)
def test_bagua_default():
    model = BoringModel()
    trainer = Trainer(max_epochs=1, strategy="bagua", gpus=1)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_gradient_allreduce():
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="gradient_allreduce")
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_bytegrad():
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="bytegrad")
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_decentralized():
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="decentralized")
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_low_prec_decentralized():
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="low_precision_decentralized")
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_async():
    model = BoringModel()
    bagua_strategy = BaguaStrategy(algorithm="async", warmup_steps=10, sync_interval_ms=50)
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_qadam():
    model = BoringModel4QAdam()
    bagua_strategy = BaguaStrategy(algorithm="qadam")
    trainer = Trainer(max_epochs=1, strategy=bagua_strategy, gpus=2)
    trainer.fit(model)


@RunIf(skip_windows=True, bagua=True, min_gpus=2)
def test_bagua_reduce():
    from pytorch_lightning.utilities.distributed import ReduceOp

    trainer = Trainer(strategy="bagua", gpus=2)
    trainer.strategy.setup_environment()

    # faster this way
    reduce_ops = [None, "mean", "AVG", "undefined", "sum", ReduceOp.SUM, ReduceOp.MAX, ReduceOp.PRODUCT, ReduceOp.MIN]
    tensor = torch.randn(10).cuda()
    for reduce_op in reduce_ops:
        if reduce_op == "undefined":
            with pytest.raises(ValueError, match="unrecognized `reduce_op`"):
                result = trainer.strategy.reduce(tensor, reduce_op=reduce_op)
        else:
            result = trainer.strategy.reduce(tensor, reduce_op=reduce_op)
