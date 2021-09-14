from unittest.mock import Mock

import pytest
from torch.optim import SGD, Adam

from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.helpers import BoringModel


@pytest.mark.parametrize(
    "frequencies,expected",
    [
        (
            (3, 1),
            [
                (0, "SGD"),
                (0, "SGD"),
                (0, "SGD"),
                (1, "Adam"),
                (0, "SGD"),
                (0, "SGD"),
                (0, "SGD"),
                (1, "Adam"),
                (0, "SGD"),
                (0, "SGD"),
            ],
        ),
        (
            (1, 2),
            [
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
            ],
        ),
    ],
)
def test_optimizer_frequencies(tmpdir, frequencies, expected):
    """Test that the optimizer loop runs optimization for the correct optimizer and optimizer idx when different
    frequencies are requested."""

    class CurrentModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            opt0 = SGD(self.parameters(), lr=0.1)
            opt1 = Adam(self.parameters(), lr=0.1)
            return {"optimizer": opt0, "frequency": frequencies[0]}, {"optimizer": opt1, "frequency": frequencies[1]}

    model = CurrentModel()
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=10,
        progress_bar_refresh_rate=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)

    positional_args = [c[0] for c in model.optimizer_step.call_args_list]
    pl_optimizer_sequence = [args[2] for args in positional_args]
    opt_idx_sequence = [args[3] for args in positional_args]
    assert all(isinstance(opt, LightningOptimizer) for opt in pl_optimizer_sequence)
    optimizer_sequence = [opt._optimizer.__class__.__name__ for opt in pl_optimizer_sequence]
    assert list(zip(opt_idx_sequence, optimizer_sequence)) == expected
