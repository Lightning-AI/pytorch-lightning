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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_metric_lightning_ddp(tmpdir):
    tutils.set_random_master_port()

    # Dummy dataset, where sum is known
    data=torch.arange(10)[:,None].float()
    dataset = torch.utils.data.TensorDataset(data)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = SumMetric()
            self.p = torch.nn.Linear(1,1) # fake params

        def training_step(self, batch, batch_idx):
            val = self.metric(batch[0])
            self.log("sum", self.metric, on_step=False, on_epoch=True)
            return self.p(val.view(1,1))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            )

        def configure_optimizers(self):
            return None

    model = TestModel()
    trainer = Trainer(
        gpus=2,
        max_epochs=1,
        log_every_n_steps=1,
        accelerator='ddp',
        progress_bar_refresh_rate=0,
        replace_sampler_ddp=False
    )
    trainer.fit(model)

    logged = trainer.logged_metrics

    assert torch.tensor(logged["sum"]) == dataset.tensors[0].sum(), \
        "Metrics did not accumulate correctly in ddp mode"



