import shutil
import tempfile
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class FakeDataset(Dataset):
    def __init__(self):
        self.data = [torch.randn(3) for _ in range(4)]
        self.labels = [torch.randint(0, 2, (1,)) for _ in range(4)]

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def save_model(model: torch.nn.Module, step_idx: int, saved_models):
    model_copy = deepcopy(model)
    state_dict = model_copy.cpu().state_dict()
    saved_models[step_idx] = state_dict


def load_model(step_idx: int, saved_models):
    return saved_models[step_idx]


class SimpleModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)
        self.automatic_optimization = False
        self.fake_losses = [
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        ]
        self.saved_models = {}

    def training_step(self, batch, batch_idx):
        out = self.layer(batch[0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch[1].float())
        self.log("loss", self.fake_losses[batch_idx], on_step=True, on_epoch=True, logger=True)
        # Save model before optimization
        save_model(self.layer, batch_idx, self.saved_models)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@contextmanager
def cleanup_after_test():
    """Context manager to ensure all test artifacts are cleaned up."""
    log_dir = Path("tests_pytorch/lightning_logs")
    try:
        yield
    finally:
        # Clean up any remaining log files
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)


def test_model_checkpoint_manual_opt():
    with cleanup_after_test(), tempfile.TemporaryDirectory() as tmpdir:
        dataset = FakeDataset()
        train_dataloader = DataLoader(dataset, batch_size=1)
        model = SimpleModule()
        trainer = Trainer(
            max_epochs=1,
            callbacks=[
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="loss",
                    dirpath=tmpdir,
                    mode="min",
                    save_last=False,
                    every_n_train_steps=1,
                    train_time_interval=None,
                    every_n_epochs=0,
                    save_on_train_epoch_end=True,
                    save_weights_only=True,
                )
            ],
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            logger=False,  # Disable logging to prevent creating lightning_logs
        )
        try:
            trainer.fit(model, train_dataloader)
        finally:
            trainer._teardown()  # Ensure trainer is properly closed

        # The best loss is at batch_idx=2 (loss=0.0)
        best_step = 2
        model_before_opt = load_model(best_step, model.saved_models)
        # Load the best checkpoint
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        best_ckpt = torch.load(best_ckpt_path, weights_only=True)["state_dict"]

        # The checkpoint should match the model before opt.step(), not after
        for layer_name, layer_value in best_ckpt.items():
            assert torch.equal(model_before_opt[layer_name.removeprefix("layer.")], layer_value.cpu()), (
                f"Mismatch in {layer_name}: checkpoint saved after optimization instead of before"
            )


def test_model_checkpoint_manual_opt_warning():
    """Test that a warning is raised when using manual optimization without saving the state."""

    class SimpleModuleNoSave(SimpleModule):
        def training_step(self, batch, batch_idx):
            out = self.layer(batch[0])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch[1].float())
            self.log("loss", self.fake_losses[batch_idx], on_step=True, on_epoch=True, logger=True)

            # Don't save the model state before optimization
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            return loss

    with cleanup_after_test(), tempfile.TemporaryDirectory() as tmpdir:
        dataset = FakeDataset()
        train_dataloader = DataLoader(dataset, batch_size=1, num_workers=0)  # Avoid num_workers warning
        model = SimpleModuleNoSave()

        # Clear any existing warnings
        warnings.filterwarnings("ignore", message=".*num_workers.*")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Always trigger warnings
            trainer = Trainer(
                max_epochs=1,
                callbacks=[
                    ModelCheckpoint(
                        save_top_k=1,
                        monitor="loss",
                        dirpath=tmpdir,
                        mode="min",
                        save_last=False,
                        every_n_train_steps=1,
                        train_time_interval=None,
                        every_n_epochs=0,
                        save_on_train_epoch_end=True,
                        save_weights_only=True,
                    )
                ],
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                logger=False,  # Disable logging to prevent creating lightning_logs
            )
            try:
                trainer.fit(model, train_dataloader)
            finally:
                trainer._teardown()

        # Find our warning in the list of warnings
        manual_opt_warnings = [
            str(warning.message)
            for warning in w
            if "Using ModelCheckpoint with manual optimization and every_n_train_steps" in str(warning.message)
        ]

        # Verify our warning was raised
        assert len(manual_opt_warnings) > 0, "Expected warning about manual optimization not found"
        assert "The checkpoint will contain the model state AFTER optimization" in manual_opt_warnings[0]
