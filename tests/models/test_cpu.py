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
import os
import platform
from distutils.version import LooseVersion

import pytest
import torch

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.step_result import TrainResult
from tests.base import EvalModelTemplate


def test_cpu_slurm_save_load(tmpdir):
    """Verify model save/load/checkpoint on CPU."""
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)
    version = logger.version

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        checkpoint_callback=ModelCheckpoint(dirpath=tmpdir),
    )
    result = trainer.fit(model)
    real_global_step = trainer.global_step

    # traning complete
    assert result == 1, 'cpu model failed to complete'

    # predict with trained model before saving
    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    x, y = batch
    x = x.view(x.size(0), -1)

    model.eval()
    pred_before_saving = model(x)

    # test HPC saving
    # simulate snapshot on slurm
    saved_filepath = trainer.checkpoint_connector.hpc_save(trainer.weights_save_path, logger)
    assert os.path.exists(saved_filepath)

    # new logger file to get meta
    logger = tutils.get_default_logger(tmpdir, version=version)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(dirpath=tmpdir),
    )
    model = EvalModelTemplate(**hparams)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_pred_same():
        assert trainer.global_step == real_global_step and trainer.global_step > 0

        # predict with loaded model to make sure answers are the same
        trainer.model.eval()
        new_pred = trainer.model(x)
        assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1

    model.on_epoch_start = assert_pred_same

    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)


def test_early_stopping_cpu_model(tmpdir):
    """Test each of the trainer options."""
    stopping = EarlyStopping(monitor='early_stop_on', min_delta=0.1)
    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[stopping],
        max_epochs=2,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        track_grad_norm=2,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif((platform.system() == "Darwin" and
                     LooseVersion(torch.__version__) < LooseVersion("1.3.0")),
                    reason="Distributed training is not supported on MacOS before Torch 1.3.0")
def test_multi_cpu_model_ddp(tmpdir):
    """Make sure DDP works."""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        gpus=None,
        num_processes=2,
        distributed_backend='ddp_cpu',
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)


def test_lbfgs_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        progress_bar_refresh_rate=0,
        weights_summary='top',
        limit_train_batches=0.2,
        limit_val_batches=0.2,
    )

    hparams = EvalModelTemplate.get_default_hparams()
    hparams.update(optimizer_name='lbfgs',
                   learning_rate=0.004)
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__lbfgs
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.25)


def test_default_logger_callbacks_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        progress_bar_refresh_rate=0,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test_without_loggers(trainer_options, model)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_running_test_after_fitting(tmpdir):
    """Verify test() on fitted model."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, thr=0.5)


def test_running_test_no_val(tmpdir):
    """Verify `test()` works on a model with no `val_loader`."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer)


def test_simple_cpu(tmpdir):
    """Verify continue training session on CPU."""
    model = EvalModelTemplate()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=20,
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'


def test_cpu_model(tmpdir):
    """Make sure model trains on CPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model, on_gpu=False)


def test_all_features_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        track_grad_norm=2,
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=2,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)


def test_tbptt_cpu_model(tmpdir):
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

    class BpttTestModel(EvalModelTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(
                pred, y_tensor.view(batch_size, truncated_bptt_steps))
            return {
                'loss': loss_val,
                'hiddens': self.test_hidden,
            }

        def training_epoch_end(self, training_step_outputs):
            training_step_outputs = training_step_outputs[0]
            assert len(training_step_outputs) == (sequence_size / truncated_bptt_steps)
            loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
            self.log('train_loss', loss)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    hparams = EvalModelTemplate.get_default_hparams()
    hparams.update(
        batch_size=batch_size,
        in_features=truncated_bptt_steps,
        hidden_dim=truncated_bptt_steps,
        out_features=truncated_bptt_steps
    )

    model = BpttTestModel(**hparams)
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'


def test_tbptt_cpu_model_result(tmpdir):
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

    class BpttTestModel(EvalModelTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(
                pred, y_tensor.view(batch_size, truncated_bptt_steps))

            result = TrainResult(loss_val, hiddens=self.test_hidden)
            return result

        def training_epoch_end(self, training_step_outputs):
            result = training_step_outputs
            assert isinstance(result, TrainResult)
            assert result.minimize.size(1) == (sequence_size / truncated_bptt_steps)

            result.minimize = result.minimize.mean()
            return result

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    hparams = EvalModelTemplate.get_default_hparams()
    hparams.update(
        batch_size=batch_size,
        in_features=truncated_bptt_steps,
        hidden_dim=truncated_bptt_steps,
        out_features=truncated_bptt_steps
    )

    model = BpttTestModel(**hparams)
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'


def test_tbptt_cpu_model_result_auto_reduce(tmpdir):
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

    class BpttTestModel(EvalModelTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(
                pred, y_tensor.view(batch_size, truncated_bptt_steps))

            result = TrainResult(loss_val, hiddens=self.test_hidden)
            return result

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    hparams = EvalModelTemplate.get_default_hparams()
    hparams.update(
        batch_size=batch_size,
        in_features=truncated_bptt_steps,
        hidden_dim=truncated_bptt_steps,
        out_features=truncated_bptt_steps
    )

    model = BpttTestModel(**hparams)
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
