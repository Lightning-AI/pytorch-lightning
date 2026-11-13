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
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection


@pytest.mark.parametrize("enable_graph", [False, True])
@pytest.mark.parametrize("on_step", [False, True])
@pytest.mark.parametrize("reduce_fx", ["mean", "sum", "min", "max"])
def test_logged_tensor_graph(enable_graph, on_step, reduce_fx):
    inputs = torch.tensor([2.0, 3.0], requires_grad=True)
    results = _ResultCollection(training=True)
    for index, batch_size in enumerate((2, 3)):
        results.log(
            "training_step",
            "objective",
            inputs[index].square(),
            on_step=True,
            on_epoch=True,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            batch_size=batch_size,
        )
    value = results.metrics(on_step=on_step)["callback"]["objective"]
    expected_values = {"mean": 7.0, "sum": 13.0, "min": 4.0, "max": 9.0}
    assert value.item() == pytest.approx(9.0 if on_step else expected_values[reduce_fx])
    assert value.requires_grad is enable_graph
    if enable_graph:
        expected_grads = {"mean": [1.6, 3.6], "sum": [4.0, 6.0], "min": [4.0, 0.0], "max": [0.0, 6.0]}
        expected = [0.0, 6.0] if on_step else expected_grads[reduce_fx]
        gradient = torch.autograd.grad(value, inputs)[0]
        torch.testing.assert_close(gradient, torch.tensor(expected))

    results.reset()
    assert not results["training_step.objective"].value.requires_grad
    assert results.metrics(on_step=on_step)["callback"] == {}
    fresh = torch.tensor(5.0, requires_grad=True)
    results.log(
        "training_step",
        "objective",
        fresh.square(),
        on_step=True,
        on_epoch=True,
        reduce_fx=reduce_fx,
        enable_graph=enable_graph,
        batch_size=4,
    )
    value = results.metrics(on_step=on_step)["callback"]["objective"]
    assert value.item() == pytest.approx(25.0)
    assert value.requires_grad is enable_graph
    if enable_graph:
        torch.testing.assert_close(torch.autograd.grad(value, fresh)[0], torch.tensor(10.0))


class _LoggedGradientModel(LightningModule):
    def __init__(self, on_epoch, enable_graph):
        super().__init__()
        self.automatic_optimization = False
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.on_epoch = on_epoch
        self.enable_graph = enable_graph
        self.gradients = []
        self.weights = []
        self.pending = []

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        loss = (self.weight * inputs).square().mean()
        self.log(
            "objective",
            loss,
            on_step=not self.on_epoch,
            on_epoch=self.on_epoch,
            enable_graph=self.enable_graph,
            logger=False,
            batch_size=len(inputs),
        )
        if self.on_epoch:
            self.pending.append((loss, len(inputs)))
        else:
            self._step(loss)

    def _step(self, reference_loss):
        value = self.trainer.callback_metrics["objective"]
        assert value.requires_grad is self.enable_graph
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(value if self.enable_graph else reference_loss)
        self.gradients.append(self.weight.grad.detach().clone())
        optimizer.step()
        self.weights.append(self.weight.detach().clone())

    def on_train_epoch_end(self):
        if self.on_epoch:
            reference_loss = sum(loss * size for loss, size in self.pending) / sum(size for _, size in self.pending)
            self._step(reference_loss)
            self.pending.clear()


@pytest.mark.parametrize("enable_graph", [False, True])
@pytest.mark.parametrize("on_epoch", [False, True])
def test_logged_loss_drives_manual_optimization(tmp_path, on_epoch, enable_graph):
    inputs = torch.tensor([1.0, 2.0, 3.0])
    model = _LoggedGradientModel(on_epoch, enable_graph)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(inputs), batch_size=2))

    reference = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([reference], lr=0.01, momentum=0.9)
    gradients, weights = [], []
    for _ in range(2):
        batches = [inputs] if on_epoch else inputs.split(2)
        for batch in batches:
            optimizer.zero_grad()
            (reference * batch).square().mean().backward()
            gradients.append(reference.grad.detach().clone())
            optimizer.step()
            weights.append(reference.detach().clone())
    torch.testing.assert_close(model.gradients, gradients)
    torch.testing.assert_close(model.weights, weights)
    torch.testing.assert_close(model.weight, reference)
    torch.testing.assert_close(
        trainer.optimizers[0].state[model.weight]["momentum_buffer"],
        optimizer.state[reference]["momentum_buffer"],
    )
