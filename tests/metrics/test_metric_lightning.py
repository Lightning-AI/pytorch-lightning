import torch
from tests.base.boring_model import BoringModel
from pytorch_lightning.metrics import Metric
from pytorch_lightning import Trainer


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
            self.metric = SumMetric()
            self.sum = 0.0

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            self.log("sum", self.metric, on_epoch=True, on_step=False)
            return self.step(x)

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
    assert torch.allclose(torch.tensor(logged["sum"]), model.sum)
