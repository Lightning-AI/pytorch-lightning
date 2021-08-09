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


@pytest.mark.parametrize("n_hidden_states", [1, 2])
def test_tbptt_cpu_model(tmpdir, n_hidden_states):
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

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training model with `{n_hidden_states}` hidden state failed with {trainer.state}"


@pytest.mark.parametrize("n_hidden_states", [1, 2])
def test_tbptt_cpu_model_lightning_module_property(tmpdir, n_hidden_states):
    """Test truncated back propagation through time works when set as a property of the LightningModule."""
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
        def __init__(self, batch_size, in_features, out_features, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None
            self.batch_size = batch_size
            self.layer = torch.nn.Linear(in_features, out_features)
            self.truncated_bptt_steps = truncated_bptt_steps

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(pred, y_tensor.view(batch_size, truncated_bptt_steps))
            return {
                "loss": loss_val,
                "hiddens": self.test_hidden,
            }

        def training_epoch_end(self, training_step_outputs):
            training_step_outputs = training_step_outputs[0]
            assert len(training_step_outputs) == (sequence_size / truncated_bptt_steps)
            loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
            self.log("train_loss", loss)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    model = BpttTestModel(batch_size=batch_size, in_features=truncated_bptt_steps, out_features=truncated_bptt_steps)
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training model with `{n_hidden_states}` hidden state failed with {trainer.state}"
