# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer


class _TwoOptimizerModel(LightningModule):
    def __init__(self, optimizer_cls, sync_grad, shared, failure=None):
        super().__init__()
        self.automatic_optimization = False
        self.first = torch.nn.Linear(2, 2)
        self.second = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.shared = torch.nn.Parameter(torch.tensor([0.5, -0.2])) if shared else None
        self.optimizer_cls = optimizer_cls
        self.sync_grad = sync_grad
        self.failure = failure
        self.losses = []
        self.gradients = []

    def configure_optimizers(self):
        first = list(self.first.parameters()) + [self.frozen]
        second = list(self.second.parameters()) + [self.frozen]
        if self.shared is not None:
            first.append(self.shared)
            second.append(self.shared)
        kwargs = {"momentum": 0.9} if self.optimizer_cls is torch.optim.SGD else {}
        return [self.optimizer_cls(first, lr=0.02, **kwargs), self.optimizer_cls(second, lr=0.02, **kwargs)]

    def training_step(self, batch, batch_idx):
        first, second = self.optimizers()
        original_flags = {p: p.requires_grad for p in self.parameters()}
        if self.failure is not None and self.current_epoch == 0 and batch_idx == 0:

            def invalid_forward():
                return self.first(batch[0].new_ones(1, 3)).sum()

            with (
                pytest.raises(RuntimeError, match="mat1 and mat2"),
                first.toggle_model(sync_grad=self.sync_grad),
            ):
                assert not self.second.weight.requires_grad
                if self.failure == "closure":
                    first.step(closure=invalid_forward)
                else:
                    invalid_forward()
            assert all(p.requires_grad == flag for p, flag in original_flags.items())
            assert self._param_requires_grad_state == {}

        for optimizer, layer in [(first, self.first), (second, self.second)]:
            with optimizer.toggle_model(sync_grad=self.sync_grad):
                optimizer.zero_grad()
                loss = layer(batch[0]).square().mean()
                if self.shared is not None:
                    loss = loss + 0.1 * self.shared.square().sum()
                self.manual_backward(loss)
                self.losses.append(loss.detach().clone())
                self.gradients.append(
                    [
                        None if p.grad is None else p.grad.detach().clone()
                        for group in optimizer.param_groups
                        for p in group["params"]
                    ]
                )
                optimizer.step()
            assert all(p.requires_grad == flag for p, flag in original_flags.items())


def _trainer(tmp_path):
    return Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def _loader():
    return DataLoader(TensorDataset(torch.arange(8, dtype=torch.float32).view(4, 2) / 8), batch_size=2)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("sync_grad", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("failure", ["forward", "closure"])
def test_training_recovers_original_flags_after_a_failed_optimizer_context(
    tmp_path, optimizer_cls, sync_grad, shared, failure
):
    torch.manual_seed(17)
    model = _TwoOptimizerModel(optimizer_cls, sync_grad, shared, failure)
    reference = deepcopy(model)
    reference.failure = None
    initial = deepcopy(model.state_dict())
    trainer = _trainer(tmp_path / "actual")
    reference_trainer = _trainer(tmp_path / "reference")
    trainer.fit(model, _loader())
    reference_trainer.fit(reference, _loader())
    torch.testing.assert_close(model.losses, reference.losses, rtol=0, atol=0)
    torch.testing.assert_close(model.gradients, reference.gradients, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    for optimizer, expected in zip(trainer.optimizers, reference_trainer.optimizers):
        torch.testing.assert_close(optimizer.state_dict(), expected.state_dict(), rtol=0, atol=0)
    assert trainer.global_step == reference_trainer.global_step == 8
    assert torch.equal(model.frozen, initial["frozen"])
    assert model.frozen.requires_grad is False
    assert not torch.equal(model.first.weight, initial["first.weight"])
    assert not torch.equal(model.second.weight, initial["second.weight"])


@pytest.mark.parametrize("error_cls", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("sync_grad", [False, True])
def test_optimizer_context_rethrows_the_same_exception_and_clears_flags(tmp_path, error_cls, sync_grad):
    error = error_cls("stop this attempt")

    class Model(_TwoOptimizerModel):
        def on_train_start(self):
            optimizer, _ = self.optimizers()
            flags = {p: p.requires_grad for p in self.parameters()}
            with pytest.raises(error_cls) as caught, optimizer.toggle_model(sync_grad=sync_grad):
                assert not self.second.weight.requires_grad
                raise error
            assert caught.value is error
            assert all(p.requires_grad == flag for p, flag in flags.items())
            assert self._param_requires_grad_state == {}

    model = Model(torch.optim.SGD, sync_grad, shared=True)
    _trainer(tmp_path).fit(model, _loader())
