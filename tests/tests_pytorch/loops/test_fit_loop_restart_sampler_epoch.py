from torch.utils.data import DataLoader, Sampler

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset


class MockSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.epoch = 0
        self.observed_epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        self.observed_epochs.append(self.epoch)
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.last_sampler = None

    def train_dataloader(self):
        dataset = RandomDataset(32, 64)
        sampler = MockSampler(dataset)
        self.last_sampler = sampler
        return DataLoader(dataset, sampler=sampler, batch_size=2)


def test_fit_loop_resume_sampler_epoch(tmp_path):
    """Test that the sampler epoch is restored before the iterator is instantiated on resume."""
    model = TestModel()

    # Train for 1 epoch and save checkpoint
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, enable_progress_bar=False, logger=False)
    trainer.fit(model)

    ckpt_path = trainer.checkpoint_callback.best_model_path

    # Resume training for epoch 2
    model2 = TestModel()
    trainer2 = Trainer(default_root_dir=tmp_path, max_epochs=2, enable_progress_bar=False, logger=False)
    trainer2.fit(model2, ckpt_path=ckpt_path)

    # The dataloader is instantiated and __iter__ is called.
    # The first observed epoch for the resumed run should be 1.
    assert model2.last_sampler.observed_epochs[0] == 1
