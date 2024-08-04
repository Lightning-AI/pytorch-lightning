from unittest.mock import patch

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


@pytest.mark.parametrize("num_steps", [1, 2, 3])
@patch("torch.Tensor.backward")
def test_backward_count_simple(torch_backward, num_steps):
    """Test that backward is called exactly once per step."""
    model = BoringModel()
    trainer = Trainer(max_steps=num_steps, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    assert torch_backward.call_count == num_steps

    torch_backward.reset_mock()

    trainer.test(model)
    assert torch_backward.call_count == 0


@patch("torch.Tensor.backward")
def test_backward_count_with_grad_accumulation(torch_backward):
    """Test that backward is called the correct number of times when accumulating gradients."""
    model = BoringModel()
    trainer = Trainer(
        max_epochs=1, limit_train_batches=6, accumulate_grad_batches=2, logger=False, enable_checkpointing=False
    )
    trainer.fit(model)
    assert torch_backward.call_count == 6

    torch_backward.reset_mock()

    trainer = Trainer(max_steps=6, accumulate_grad_batches=2, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    assert torch_backward.call_count == 12


@patch("torch.Tensor.backward")
def test_backward_count_with_closure(torch_backward, tmp_path):
    """Using a closure (e.g. with LBFGS) should lead to no extra backward calls."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.LBFGS(self.parameters(), lr=0.1)

    model = TestModel()
    trainer = Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    assert torch_backward.call_count == 5

    torch_backward.reset_mock()

    trainer = Trainer(max_steps=5, accumulate_grad_batches=2, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    assert torch_backward.call_count == 10
