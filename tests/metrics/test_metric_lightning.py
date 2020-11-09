import pytest

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.metrics import Metric
from tests.base.boring_model import BoringModel
import tests.base.develop_utils as tutils


class SumMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("x", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


def test_metric_lightning(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()

            return self.step(x)

        def training_epoch_end(self, outs):
            assert torch.allclose(self.sum, self.metric.compute())
            self.sum = 0.0

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)


def test_metric_lightning_log(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric_step = SumMetric()
            self.metric_epoch = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric_step(x.sum())
            self.sum += x.sum()
            self.log("sum_step", self.metric_step, on_epoch=True, on_step=False)
            return {'loss': self.step(x), 'data': x}

        def training_epoch_end(self, outs):
            self.log("sum_epoch", self.metric_epoch(torch.stack([o['data'] for o in outs]).sum()))

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged = trainer.logged_metrics
    assert torch.allclose(torch.tensor(logged["sum_step"]), model.sum)
    assert torch.allclose(torch.tensor(logged["sum_epoch"]), model.sum)


def test_scriptable(tmpdir):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            # the metric is not used in the module's `forward`
            # so the module should be exportable to TorchScript
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            self.log("sum", self.metric, on_epoch=True, on_step=False)
            return self.step(x)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)
    rand_input = torch.randn(10, 32)

    script_model = model.to_torchscript()

    # test that we can still do inference
    output = model(rand_input)
    script_output = script_model(rand_input)
    assert torch.allclose(output, script_output)
