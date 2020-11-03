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
"""
Tests to ensure that the training loop works with a dict (1.0)
"""
import os
import torch
import pytest
from copy import deepcopy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.epoch_result_store import EpochResultStore
from tests.base.boring_model import BoringModel, RandomDataset


class Helper:
    def decorator_with_arguments(fx_name='', hook_fx_name=None):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                # Set information
                self._current_fx_name = fx_name
                self._current_hook_fx_name = hook_fx_name
                self._results = Result()

                result = func(self, *args, **kwargs)

                # cache metrics
                self.trainer.logger_connector.cache_logged_metrics()
                return result
            return wrapper

        return decorator


def test__logger_connector__epoch_result_store__train(tmpdir):
    """
    Tests that LoggerConnector will properly capture logged information
    and reduce them
    """

    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):

        train_losses = []

        @Helper.decorator_with_arguments(fx_name="training_step")
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)

            self.train_losses.append(loss)

            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return {"loss": loss}

        def on_train_epoch_end(self, outputs):
            # save objects as it will be reset at the end of epoch.
            self.train_results = deepcopy(self.trainer.logger_connector.cached_results)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=4,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    train_results = model.train_results

    assert len(train_results(fx_name="training_step", dl_idx="0", opt_idx="0")) == 2
    generated = train_results(fx_name="training_step",
                              dl_idx="0",
                              opt_idx="0",
                              batch_idx="0",
                              split_idx="0")["train_loss"]
    assert generated == model.train_losses[0]
    generated = train_results(fx_name="training_step",
                              dl_idx="0",
                              opt_idx="0",
                              batch_idx="1",
                              split_idx="0")["train_loss"]
    assert generated == model.train_losses[1]

    assert train_results.has_reduced is not True

    train_results.has_batch_loop_finished = True

    assert train_results.has_reduced is True

    generated = train_results(fx_name="training_step", dl_idx="0", opt_idx="0", reduced=True)['train_loss_epoch'].item()
    excepted = torch.stack(model.train_losses).mean().item()
    assert generated == excepted


def test__logger_connector__epoch_result_store__train__ttbt(tmpdir):
    """
    Tests that LoggerConnector will properly capture logged information with ttbt
    and reduce them
    """
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

    class TestModel(BoringModel):

        train_losses = []

        def __init__(self):
            super().__init__()
            self.test_hidden = None
            self.layer = torch.nn.Linear(2, 2)

        @Helper.decorator_with_arguments(fx_name="training_step")
        def training_step(self, batch, batch_idx, hiddens):
            try:
                assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            except Exception as e:
                print(e)

            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss = torch.nn.functional.mse_loss(
                pred, y_tensor.view(batch_size, truncated_bptt_steps))

            self.train_losses.append(loss)

            self.log('a', loss, on_epoch=True)

            return {'loss': loss, 'hiddens': self.test_hidden}

        def on_train_epoch_start(self) -> None:
            self.test_hidden = None

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

        def on_train_epoch_end(self, outputs):
            # save objects as it will be reset at the end of epoch.
            self.train_results = deepcopy(self.trainer.logger_connector.cached_results)

    model = TestModel()
    model.training_epoch_end = None
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=0,
        truncated_bptt_steps=truncated_bptt_steps,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    train_results = model.train_results

    generated = train_results(fx_name="training_step", dl_idx="0", opt_idx="0", batch_idx="0")
    assert len(generated) == len(model.train_losses)

    # assert reduction didn't happen yet
    assert train_results.has_reduced is False

    # Launch reduction
    train_results.has_batch_loop_finished = True

    # assert reduction did happen
    assert train_results.has_reduced is True

    generated = train_results(fx_name="training_step", dl_idx="0", opt_idx="0", reduced=True)['a_epoch'].item()
    assert generated == torch.stack(model.train_losses).mean().item()


@pytest.mark.parametrize('num_dataloaders', [1, 2])
def test__logger_connector__epoch_result_store__test_multi_dataloaders(tmpdir, num_dataloaders):
    """
    Tests that LoggerConnector will properly capture logged information in multi_dataloaders scenario
    """

    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):

        test_losses = {}

        @Helper.decorator_with_arguments(fx_name="test_step")
        def test_step(self, batch, batch_idx, dl_idx=0):
            output = self.layer(batch)
            loss = self.loss(batch, output)

            primary_key = str(dl_idx)
            if primary_key not in self.test_losses:
                self.test_losses[primary_key] = []

            self.test_losses[primary_key].append(loss)

            self.log("test_loss", loss, on_step=True, on_epoch=True)
            return {"test_loss": loss}

        def test_dataloader(self):
            return [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(num_dataloaders)]

    model = TestModel()
    model.val_dataloader = None
    model.test_epoch_end = None

    limit_test_batches = 4

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=limit_test_batches,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.test(model)

    test_results = trainer.logger_connector._cached_results["test"]

    generated = test_results(fx_name="test_step")
    assert len(generated) == num_dataloaders

    for dl_idx in range(num_dataloaders):
        generated = len(test_results(fx_name="test_step", dl_idx=str(dl_idx)))
        assert generated == limit_test_batches

    test_results.has_batch_loop_finished = True

    for dl_idx in range(num_dataloaders):
        expected = torch.stack(model.test_losses[str(dl_idx)]).mean()
        generated = test_results(fx_name="test_step", dl_idx=str(dl_idx), reduced=True)["test_loss_epoch"]
        assert abs(expected.item() - generated.item()) < 1e-6
