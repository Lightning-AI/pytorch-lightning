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

import torch

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


def test_cpu_slurm_save_load(tmpdir):
    """Verify model save/load/checkpoint on CPU."""
    model = BoringModel()

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
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)
    real_global_step = trainer.global_step

    # traning complete
    assert trainer.state.finished, 'cpu model failed to complete'

    # predict with trained model before saving
    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    model.eval()
    pred_before_saving = model(batch)

    # test HPC saving
    # simulate snapshot on slurm
    saved_filepath = trainer.checkpoint_connector.hpc_save(trainer.weights_save_path, logger)
    assert os.path.exists(saved_filepath)

    # new logger file to get meta
    logger = tutils.get_default_logger(tmpdir, version=version)

    model = BoringModel()

    class _StartCallback(Callback):
        # set the epoch start hook so we can predict before the model does the full training
        def on_train_epoch_start(self, trainer, model):
            assert trainer.global_step == real_global_step and trainer.global_step > 0
            # predict with loaded model to make sure answers are the same
            mode = model.training
            model.eval()
            new_pred = model(batch)
            assert torch.eq(pred_before_saving, new_pred).all()
            model.train(mode)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        callbacks=[_StartCallback(), ModelCheckpoint(dirpath=tmpdir)],
    )
    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)


def test_early_stopping_cpu_model(tmpdir):

    class ModelTrainVal(BoringModel):

        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log('val_loss', output['x'])
            return output

    tutils.reset_seed()
    stopping = EarlyStopping(monitor="val_loss", min_delta=0.1)
    trainer_options = dict(
        callbacks=[stopping],
        default_root_dir=tmpdir,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        track_grad_norm=2,
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=2,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    model = ModelTrainVal()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


@RunIf(skip_windows=True)
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
        accelerator='ddp_cpu',
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, data=dm, on_gpu=False)


def test_lbfgs_cpu_model(tmpdir):
    """Test each of the trainer options. Testing LBFGS optimizer"""

    class ModelSpecifiedOptimizer(BoringModel):

        def __init__(self, optimizer_name, learning_rate):
            super().__init__()
            self.optimizer_name = optimizer_name
            self.learning_rate = learning_rate
            self.save_hyperparameters()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        progress_bar_refresh_rate=0,
        weights_summary="top",
        limit_train_batches=0.2,
        limit_val_batches=0.2,
    )

    model = ModelSpecifiedOptimizer(optimizer_name="LBFGS", learning_rate=0.004)
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)


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

    model = BoringModel()
    tpipes.run_model_test_without_loggers(trainer_options, model, min_acc=0.01)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_running_test_after_fitting(tmpdir):
    """Verify test() on fitted model."""

    class ModelTrainValTest(BoringModel):

        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args, **kwargs)
            self.log('val_loss', output['x'])
            return output

        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log('test_loss', output['y'])
            return output

    model = ModelTrainValTest()

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
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key='test_loss', thr=0.5)


def test_running_test_no_val(tmpdir):
    """Verify `test()` works on a model with no `val_dataloader`. It performs
    train and test only"""

    class ModelTrainTest(BoringModel):

        def val_dataloader(self):
            pass

        def test_step(self, *args, **kwargs):
            output = super().test_step(*args, **kwargs)
            self.log('test_loss', output['y'])
            return output

    model = ModelTrainTest()

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
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer, key='test_loss')


def test_simple_cpu(tmpdir):
    """Verify continue training session on CPU."""
    model = BoringModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=20,
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state.finished, 'amp + ddp model failed to complete'


def test_cpu_model(tmpdir):
    """Make sure model trains on CPU."""
    trainer_options = dict(
        default_root_dir=tmpdir, progress_bar_refresh_rate=0, max_epochs=1, limit_train_batches=4, limit_val_batches=4
    )

    model = BoringModel()
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
        limit_val_batches=0.4,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model, on_gpu=False, min_acc=0.01)


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

    class BpttTestModel(BoringModel):

        def __init__(self, batch_size, in_features, out_features, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_hidden = None
            self.batch_size = batch_size
            self.layer = torch.nn.Linear(in_features, out_features)

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
        truncated_bptt_steps=truncated_bptt_steps,
        limit_val_batches=0,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
