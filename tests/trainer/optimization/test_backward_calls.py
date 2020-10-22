from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


@pytest.mark.parametrize("num_steps", [1, 2, 3])
@patch("torch.Tensor.backward")
def test_backward_count_simple(torch_backward, num_steps):
    """ Test that backward is called exactly once per step. """
    model = EvalModelTemplate()
    trainer = Trainer(max_steps=num_steps)
    trainer.fit(model)
    assert torch_backward.call_count == num_steps

    torch_backward.reset_mock()

    trainer.test(model)
    assert torch_backward.call_count == 0


@patch("torch.Tensor.backward")
def test_backward_count_with_grad_accumulation(torch_backward):
    """ Test that backward is called the correct number of times when accumulating gradients. """
    model = EvalModelTemplate()
    trainer = Trainer(max_epochs=1, limit_train_batches=6, accumulate_grad_batches=2)
    trainer.fit(model)
    assert torch_backward.call_count == 6

    torch_backward.reset_mock()

    trainer = Trainer(max_steps=6, accumulate_grad_batches=2)
    trainer.fit(model)
    assert torch_backward.call_count == 12


@patch("torch.Tensor.backward")
def test_backward_count_with_closure(torch_backward):
    """ Using a closure (e.g. with LBFGS) should lead to no extra backward calls. """
    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__lbfgs
    trainer = Trainer(max_steps=5)
    trainer.fit(model)
    assert torch_backward.call_count == 5

    torch_backward.reset_mock()

    trainer = Trainer(max_steps=5, accumulate_grad_batches=2)
    trainer.fit(model)
    assert torch_backward.call_count == 10
