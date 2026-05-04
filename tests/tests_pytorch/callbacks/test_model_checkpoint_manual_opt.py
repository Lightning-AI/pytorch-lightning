import shutil
import tempfile
import warnings
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Callback, LightningModule, Trainer
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


def test_model_checkpoint_manual_opt_train_time_interval():
    """Regression: ``train_time_interval`` must fire mid-run under manual optimization.

    Before the fix, the manual-optimization branch in ``on_train_batch_end`` only
    inspected ``every_n_train_steps`` and silently no-op'd when ``train_time_interval``
    was the only configured trigger. ``last.ckpt`` was still written by ``on_train_end``,
    so end-of-run state checks miss the bug -- this test asserts the mid-run save by
    observing ``_last_global_step_saved`` from a spy callback queued after the
    ``ModelCheckpoint``.
    """
    saved_steps_during_training = []

    class _Spy(Callback):
        def __init__(self, ckpt: ModelCheckpoint) -> None:
            self.ckpt = ckpt

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            saved_steps_during_training.append(self.ckpt._last_global_step_saved)

    with cleanup_after_test(), tempfile.TemporaryDirectory() as tmpdir:
        dataset = FakeDataset()
        train_dataloader = DataLoader(dataset, batch_size=1)
        model = SimpleModule()
        ckpt = ModelCheckpoint(
            dirpath=tmpdir,
            save_top_k=0,
            save_last=True,
            train_time_interval=timedelta(seconds=0),
            save_weights_only=True,
        )
        trainer = Trainer(
            max_epochs=1,
            callbacks=[ckpt, _Spy(ckpt)],
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            logger=False,
        )
        try:
            trainer.fit(model, train_dataloader)
        finally:
            trainer._teardown()

        # With ``train_time_interval=0``, the callback must fire on every batch.
        # Pre-fix the value stayed at 0 until ``on_train_end`` saved once.
        assert any(step > 0 for step in saved_steps_during_training), (
            "ModelCheckpoint(train_time_interval=...) silently no-op'd mid-run under manual_optimization; "
            f"observed _last_global_step_saved values during training: {saved_steps_during_training}"
        )
        assert (Path(tmpdir) / "last.ckpt").exists()


def test_model_checkpoint_manual_opt_broadcasts_skip_time_unconditionally():
    """All ranks must agree on ``skip_time`` even when ``train_time_interval`` is unset.

    The manual-opt branch broadcasts ``skip_time`` from rank 0 so a future divergence
    in the skip-time path cannot leave some ranks blocking on a collective alone.

    """
    broadcasted: list[bool] = []

    with cleanup_after_test(), tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            max_epochs=1,
            callbacks=[ModelCheckpoint(dirpath=tmpdir, save_top_k=0, save_last=True, save_weights_only=True)],
            num_sanity_val_steps=0,
            logger=False,
        )
        original_broadcast = trainer.strategy.broadcast

        def _spy(obj, src=0):
            if isinstance(obj, bool):
                broadcasted.append(obj)
            return original_broadcast(obj, src)

        trainer.strategy.broadcast = _spy
        try:
            trainer.fit(SimpleModule(), DataLoader(FakeDataset(), batch_size=1))
        finally:
            trainer._teardown()

    # 4 training batches; pre-fix this branch broadcasts 0 times when train_time_interval is None.
    assert len(broadcasted) >= len(FakeDataset()), (
        f"expected one broadcast per train batch, got {len(broadcasted)}: {broadcasted}"
    )
