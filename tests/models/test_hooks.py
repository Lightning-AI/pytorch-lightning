import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('max_steps', [1, 2, 3])
def test_on_before_zero_grad_called(max_steps):

    class CurrentTestModel(EvalModelTemplate):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel()

    trainer = Trainer(
        max_steps=max_steps,
        num_sanity_val_steps=5,
    )
    assert 0 == model.on_before_zero_grad_called
    trainer.fit(model)
    assert max_steps == model.on_before_zero_grad_called

    model.on_before_zero_grad_called = 0
    trainer.test(model)
    assert 0 == model.on_before_zero_grad_called
