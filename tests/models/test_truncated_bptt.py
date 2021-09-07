# Copyright The PyTorch Lightning team.
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

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


class LinearModel(BoringModel):
    """Linear model for testing TBPTT with automatic optimization."""

    def __init__(self, truncated_bptt_steps=2, n_hidden_states=1, sequence_size=30, batch_size=30):
        super().__init__()
        self.truncated_bptt_steps = truncated_bptt_steps
        self.n_hidden_states = n_hidden_states
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.automatic_optimization = True

        self.example_input_array = torch.randn(5, truncated_bptt_steps)
        self.layer = torch.nn.Linear(in_features=truncated_bptt_steps, out_features=truncated_bptt_steps)
        self.test_hidden = None

    def training_step(self, batch, batch_idx, hiddens):
        assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
        if self.n_hidden_states == 1:
            self.test_hidden = torch.rand(1)
        else:
            self.test_hidden = tuple([torch.rand(1)] * self.n_hidden_states)

        x_tensor, y_list = batch
        assert x_tensor.shape[1] == self.truncated_bptt_steps, "tbptt split Tensor failed"

        y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
        assert y_tensor.shape[1] == self.truncated_bptt_steps, "tbptt split list failed"

        pred = self(x_tensor.view(self.batch_size, self.truncated_bptt_steps))
        loss_val = torch.nn.functional.mse_loss(pred, y_tensor.view(self.batch_size, self.truncated_bptt_steps))
        return {"loss": loss_val, "hiddens": self.test_hidden}

    def training_epoch_end(self, training_step_outputs):
        training_step_outputs = training_step_outputs[0]
        assert len(training_step_outputs) == (self.sequence_size / self.truncated_bptt_steps)
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        assert loss.grad_fn is None
        self.log("train_loss", loss)


class ManualLinearModel(LinearModel):
    """Linear model for testing TBPTT with manual optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx, hiddens):
        out = super().training_step(batch, batch_idx, hiddens)
        loss, hiddens = out["loss"], out["hiddens"]
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        assert loss.grad_fn is not None
        return {"loss": loss, "hiddens": hiddens}


@pytest.mark.parametrize("model_class", (LinearModel, ManualLinearModel))
@pytest.mark.parametrize("n_hidden_states", (1, 2))
def test_tbptt_cpu_model(tmpdir, n_hidden_states, model_class):
    """Test truncated back propagation through time works with automatic and manual optimization."""

    sequence_size = 30
    batch_size = 30

    seq2seq_dataset = TensorDataset(
        torch.rand(batch_size, sequence_size, 1),
        torch.rand(batch_size, sequence_size, 1),
    )

    train_dataloader = DataLoader(dataset=seq2seq_dataset, batch_size=batch_size, shuffle=False)
    model = model_class(n_hidden_states=n_hidden_states, sequence_size=sequence_size, batch_size=batch_size)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0,
        weights_summary=None,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)


def test_tbptt_log(tmpdir):
    truncated_bptt_steps = 2
    N, T, F = 32, 15, 1  # batches x timesteps (sequence size) x features
    batch_size = 10
    assert T % truncated_bptt_steps != 0, "Should test leftover time steps"

    seq2seq_dataset = TensorDataset(torch.rand(N, T, F), torch.rand(N, T, F))
    train_dataloader = DataLoader(dataset=seq2seq_dataset, batch_size=batch_size)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.test_hidden = None
            self.layer = torch.nn.LSTM(input_size=F, hidden_size=T, batch_first=True)
            self.truncated_bptt_steps = truncated_bptt_steps

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            if hiddens is not None:
                assert hiddens.grad_fn is None
            split_idx = self.trainer.fit_loop.split_idx
            self.test_hidden = torch.tensor(split_idx, requires_grad=True, dtype=torch.float).pow(2)

            x, y = batch
            if self.trainer.fit_loop.epoch_loop.batch_loop.done:
                # last split idx, not aligned
                assert x.shape[1] == T % truncated_bptt_steps
                assert y.shape[1] == T % truncated_bptt_steps
            else:
                assert x.shape[1] == truncated_bptt_steps
                assert y.shape[1] == truncated_bptt_steps

            pred, _ = self(x)
            loss = torch.nn.functional.mse_loss(pred, y)

            self.log("a", loss, on_epoch=True)

            return {"loss": loss, "hiddens": self.test_hidden}

        def on_train_batch_start(self, *args, **kwargs) -> None:
            self.test_hidden = None

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=2,
        weights_summary=None,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

    assert trainer.fit_loop.batch_idx == N // batch_size
    assert trainer.fit_loop.split_idx == T // truncated_bptt_steps
    assert set(trainer.logged_metrics) == {"a_step", "a_epoch", "epoch"}
