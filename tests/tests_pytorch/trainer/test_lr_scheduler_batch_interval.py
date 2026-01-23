"""Tests for batch interval learning rate scheduler support."""

from torch import optim
from torch.optim.lr_scheduler import StepLR

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class LRTrackerModule(BoringModel):
    """Module that tracks learning rates for testing."""

    def __init__(self, *args, scheduler_interval="epoch", scheduler_frequency=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.lr_history = []
        self.step_history = []

    def on_train_batch_start(self, batch, batch_idx):
        """Track current LR at each batch start."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.lr_history.append(lr)
        self.step_history.append(self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            },
        }


def test_batch_interval_scheduler_updates():
    """Test that batch interval schedulers update on every batch."""
    model = LRTrackerModule(scheduler_interval="batch")
    trainer = Trainer(
        max_epochs=1, limit_train_batches=10, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)

    # With batch interval and StepLR(step_size=1), LR should decrease every batch
    assert len(model.lr_history) == 10
    # LR should decrease over batches
    assert model.lr_history[0] > model.lr_history[-1], "LR should decrease with batch interval"
    # Each batch should have a different LR (with gamma=0.5)
    assert model.lr_history[0] > model.lr_history[1]
    assert model.lr_history[1] > model.lr_history[2]


def test_step_interval_scheduler_updates():
    """Test that step interval schedulers still work correctly."""
    model = LRTrackerModule(scheduler_interval="step")
    trainer = Trainer(
        max_epochs=1, limit_train_batches=10, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)

    # With step interval and no gradient accumulation, behaves like batch interval
    assert len(model.lr_history) == 10


def test_epoch_interval_scheduler_updates():
    """Test that epoch interval schedulers only update once per epoch."""
    model = LRTrackerModule(scheduler_interval="epoch")
    trainer = Trainer(
        max_epochs=3, limit_train_batches=10, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)

    # With epoch interval, LR should be constant within an epoch but change between epochs
    lr_history = model.lr_history
    epoch_size = 10

    # All LRs in first epoch should be the same
    assert all(lr == lr_history[0] for lr in lr_history[:epoch_size])
    # All LRs in second epoch should be the same but different from first
    assert all(lr == lr_history[epoch_size] for lr in lr_history[epoch_size : 2 * epoch_size])
    # LRs should decrease from epoch to epoch
    assert lr_history[0] > lr_history[epoch_size]
    assert lr_history[epoch_size] > lr_history[2 * epoch_size]


def test_batch_interval_with_frequency():
    """Test batch interval with frequency > 1."""
    model = LRTrackerModule(scheduler_interval="batch", scheduler_frequency=2)
    trainer = Trainer(
        max_epochs=1, limit_train_batches=10, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)

    # With frequency=2, LR should update every 2 batches
    # Initially LR=1.0
    # After batch 2 (idx 1): LR=0.5
    # After batch 4 (idx 3): LR=0.25
    # etc.
    assert len(model.lr_history) == 10
    assert model.lr_history[0] == 1.0  # batch 0
    assert model.lr_history[1] == 1.0  # batch 1, no update yet
    assert model.lr_history[2] < 1.0  # batch 2, update happened
    # More batches should have lower LR as we progress
    assert model.lr_history[-1] < model.lr_history[0]


def test_batch_interval_with_gradient_accumulation():
    """Test batch interval with gradient accumulation."""
    model = LRTrackerModule(scheduler_interval="batch")
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=0,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)

    # Even with gradient accumulation, batch interval should update every batch
    # (not every optimizer step)
    assert len(model.lr_history) == 10
    # LR should still decrease
    assert model.lr_history[0] > model.lr_history[-1]


def test_mixed_intervals():
    """Test that batch, step, and epoch intervals work together."""

    class MixedSchedulerModule(BoringModel):
        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=1.0)
            scheduler1 = StepLR(optimizer, step_size=1, gamma=0.5)
            scheduler2 = StepLR(optimizer, step_size=1, gamma=0.9)
            return [optimizer], [
                {"scheduler": scheduler1, "interval": "batch", "frequency": 1},
                {"scheduler": scheduler2, "interval": "epoch", "frequency": 1},
            ]

    model = MixedSchedulerModule()
    trainer = Trainer(
        max_epochs=2, limit_train_batches=5, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    # This should not raise an error
    trainer.fit(model)


def test_batch_interval_initialization():
    """Test that batch interval schedulers are properly initialized."""

    class InitTestModule(LRTrackerModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, scheduler_interval="batch", **kwargs)
            self.initial_lr_checked = False

        def on_train_start(self):
            # Check that initial LR is correct
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            assert lr == 1.0, f"Expected initial LR of 1.0, got {lr}"
            self.initial_lr_checked = True

    model = InitTestModule()
    trainer = Trainer(
        max_epochs=1, limit_train_batches=2, limit_val_batches=0, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)
    assert model.initial_lr_checked


if __name__ == "__main__":
    test_batch_interval_scheduler_updates()
    test_step_interval_scheduler_updates()
    test_epoch_interval_scheduler_updates()
    test_batch_interval_with_frequency()
    test_batch_interval_with_gradient_accumulation()
    test_mixed_intervals()
    test_batch_interval_initialization()
    print("All tests passed!")
