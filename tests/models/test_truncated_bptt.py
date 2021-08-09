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

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


@pytest.mark.parametrize("n_hidden_states", (1, 2))
@pytest.mark.parametrize("property_on_module", (False, True))
def test_tbptt_cpu_model(tmpdir, n_hidden_states, property_on_module):
    """Test truncated back propagation through time works."""
    truncated_bptt_steps = 2
    sequence_size = 30
    batch_size = 30

    x_seq = torch.rand(batch_size, sequence_size, 1)
    y_seq_list = torch.rand(batch_size, sequence_size, 1).tolist()

    class MockSeq2SeqDataset(torch.utils.data.Dataset):
        def __getitem__(self, i):
            return x_seq, y_seq_list

        def __len__(self):
            return 1

    class BpttTestModel(BoringModel):
        def __init__(self, batch_size, in_features, out_features, n_hidden_states, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None
            self.batch_size = batch_size
            self.layer = torch.nn.Linear(in_features, out_features)
            self.n_hidden_states = n_hidden_states
            if property_on_module:
                self.truncated_bptt_steps = truncated_bptt_steps

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            if self.n_hidden_states == 1:
                self.test_hidden = torch.rand(1)
            else:
                self.test_hidden = tuple([torch.rand(1)] * self.n_hidden_states)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(pred, y_tensor.view(batch_size, truncated_bptt_steps))
            return {"loss": loss_val, "hiddens": self.test_hidden}

        def training_epoch_end(self, training_step_outputs):
            training_step_outputs = training_step_outputs[0]
            assert len(training_step_outputs) == (sequence_size / truncated_bptt_steps)
            loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
            self.log("train_loss", loss)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(), batch_size=batch_size, shuffle=False, sampler=None
            )

    model = BpttTestModel(
        batch_size=batch_size,
        in_features=truncated_bptt_steps,
        out_features=truncated_bptt_steps,
        n_hidden_states=n_hidden_states,
    )
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    trainer_tbptt_steps = None if property_on_module else truncated_bptt_steps

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=trainer_tbptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training model with `{n_hidden_states}` hidden state failed with {trainer.state}"


def test_tbptt_log(tmpdir):
    truncated_bptt_steps = 2
    N, T, F = 32, 15, 1  # batches x timesteps (sequence size) x features
    batch_size = 10
    assert T % truncated_bptt_steps != 0, "Should test leftover time steps"

    class MockSeq2SeqDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x_seq = torch.randn(N, T, F)
            self.y_seq = torch.randn(N, T, F)

        def __getitem__(self, index):
            return self.x_seq[index], self.y_seq[index]

        def __len__(self):
            return N

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

        def train_dataloader(self):
            return torch.utils.data.DataLoader(dataset=MockSeq2SeqDataset(), batch_size=batch_size)

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=2,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.fit_loop.batch_idx == N // batch_size
    assert trainer.fit_loop.split_idx == T // truncated_bptt_steps
    assert set(trainer.logged_metrics) == {"a_step", "a_epoch", "epoch"}
