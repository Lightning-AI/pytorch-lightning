from unittest.mock import Mock

from torch.optim import SGD, Adam

from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.helpers import BoringModel


def test_optimizer_frequencies(tmpdir):
    """Test that the optimizer loop runs optimization for the correct optimizer and optimizer idx when frequencies
    when different frequencies are requested."""
    # call first optimizer 3 times, then second optimizer 1 time, then first optimizer 3 times, etc.
    freq = (3, 1)

    class CurrentModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            opt0 = SGD(self.parameters(), lr=0.1)
            opt1 = Adam(self.parameters(), lr=0.1)
            return {"optimizer": opt0, "frequency": freq[0]}, {"optimizer": opt1, "frequency": freq[1]}

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
    assert optimizer_sequence == ["SGD", "SGD", "SGD", "Adam", "SGD", "SGD", "SGD", "Adam", "SGD", "SGD"]
    assert opt_idx_sequence == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
