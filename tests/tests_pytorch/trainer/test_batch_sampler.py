import pytest
from torch.utils.data import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch import Callback, Trainer, seed_everything
from tests_pytorch.helpers.runif import RunIf
from lightning.pytorch.demos.boring_classes import (
    BoringModel,
    RandomDataset,
)


class DistribBatchSamplerCallback(Callback):
    def __init__(self, expected_batch_size, expected_drop_last):
        self.expected_batch_size = expected_batch_size
        self.expected_drop_last = expected_drop_last

    def on_train_start(self, trainer, pl_module):
        assert isinstance(trainer.train_dataloader.sampler, DistributedSampler)
        assert trainer.train_dataloader.batch_size == self.expected_batch_size
        assert trainer.train_dataloader.drop_last == self.expected_drop_last


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("drop_last", [False, True])
@RunIf(min_cuda_gpus=2, skip_windows=True)
def test_dataloader_distributed_batch_sampler(tmp_path, batch_size, drop_last):
    """Test BatchSampler and it's arguments for DDP backend."""
    seed_everything(123)
    dataset = RandomDataset(32, 64)
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    print(batch_sampler.drop_last, dataloader.drop_last)
    model = BoringModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1],
        num_nodes=1,
        strategy="ddp",
        default_root_dir=tmp_path,
        max_steps=1,
        callbacks=[DistribBatchSamplerCallback(expected_batch_size=batch_size, expected_drop_last=drop_last)],
    )
    trainer.fit(model, train_dataloaders=dataloader)
