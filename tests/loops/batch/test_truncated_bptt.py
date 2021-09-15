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
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer


class LSTMModel(LightningModule):
    """LSTM sequence-to-sequence model for testing TBPTT with automatic optimization."""

    def __init__(self, truncated_bptt_steps=2, input_size=1, hidden_size=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.automatic_optimization = True

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx, hiddens):
        x, y = batch
        pred, hiddens = self.lstm(x, hiddens)
        loss = F.mse_loss(pred, y)
        return {"loss": loss, "hiddens": hiddens}

    def train_dataloader(self):
        dataset = TensorDataset(torch.rand(16, 8, self.input_size), torch.rand(16, 8, self.input_size))
        return DataLoader(dataset=dataset, batch_size=4)


class ManualLSTMModel(LSTMModel):
    """LSTM sequence-to-sequence model for testing TBPTT with manual optimization."""

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
        return {"loss": loss, "hiddens": hiddens}


@pytest.mark.parametrize("model_class", (LSTMModel, ManualLSTMModel))
def test_persistent_hidden_state_transfer(tmpdir, model_class):
    """Test that the hidden state reference gets passed through from one training_step to the next and remains
    unmodified apart from detached grad_fn."""

    class TBPTTModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None

        def training_step(self, batch, batch_idx, hiddens):
            split_idx = self.trainer.fit_loop.split_idx
            # the hidden state may only be None for the first split_idx
            assert not ((split_idx == 0) ^ (hiddens is None))
            # test_hiddens is None when hiddens is None
            assert not ((hiddens is None) ^ (self.test_hidden is None))
            # the states are equal (persistent)
            assert hiddens is None or all(torch.equal(h, th) for h, th in zip(hiddens, self.test_hidden))
            # the incoming hidden state never has a grad_fn (gets automatically detached)
            assert hiddens is None or all(h.grad_fn is None for h in hiddens)
            out = super().training_step(batch, batch_idx, hiddens)

            # store hiddens, assert persistence in next training_step
            self.test_hidden = out["hiddens"]

            # hiddens may have grad_fn when returning, gets automatically detached
            assert all(h.grad_fn is not None for h in self.test_hidden)
            return out

        def on_train_batch_start(self, *_, **__) -> None:
            self.test_hidden = None

    model = TBPTTModel(truncated_bptt_steps=2, input_size=1, hidden_size=8)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize("model_class", (LSTMModel, ManualLSTMModel))
def test_tbptt_split_shapes(tmpdir, model_class):
    """Test that the sequence data gets split correctly and that the outputs are correctly passed from hook to
    hook."""
    batch_size = 10
    truncated_bptt_steps = 2
    n, t, f = 32, 15, 1  # (num samples, sequence size, input size)
    assert t % truncated_bptt_steps != 0, "test must run with sequence length not divisible by tbptt steps"

    seq2seq_dataset = TensorDataset(torch.rand(n, t, f), torch.rand(n, t, f))
    train_dataloader = DataLoader(dataset=seq2seq_dataset, batch_size=batch_size)

    class TBPTTModel(model_class):
        def training_step(self, batch, batch_idx, hiddens):
            x, y = batch
            if self.trainer.fit_loop.epoch_loop.batch_loop.done:
                # last split idx, not aligned
                assert x.shape[1] == t % truncated_bptt_steps
                assert y.shape[1] == t % truncated_bptt_steps
            else:
                assert x.shape[1] == truncated_bptt_steps
                assert y.shape[1] == truncated_bptt_steps
            return super().training_step(batch, batch_idx, hiddens)

        def training_epoch_end(self, training_step_outputs):
            training_step_outputs = training_step_outputs[0]
            assert len(training_step_outputs) == math.ceil(t / self.truncated_bptt_steps)
            assert all(out["loss"].grad_fn is None for out in training_step_outputs)
            assert all("hiddens" not in out for out in training_step_outputs)

    model = TBPTTModel(truncated_bptt_steps=truncated_bptt_steps, input_size=f, hidden_size=8)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

    assert trainer.fit_loop.batch_idx == n // batch_size
    assert trainer.fit_loop.split_idx == t // truncated_bptt_steps


@pytest.mark.parametrize("model_class", (LSTMModel, ManualLSTMModel))
def test_tbptt_logging(tmpdir, model_class):
    """Test step-level and epoch-level logging works with TBPTT."""

    class TBPTTModel(model_class):
        def training_step(self, *args, **kwargs):
            out = super().training_step(*args, **kwargs)
            self.log("loss", out["loss"], on_step=True, on_epoch=True)
            return out

    model = TBPTTModel(truncated_bptt_steps=2)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        log_every_n_steps=2,
        weights_summary=None,
        checkpoint_callback=False,
    )
    trainer.fit(model)
    assert set(trainer.logged_metrics) == {"loss_step", "loss_epoch", "epoch"}
