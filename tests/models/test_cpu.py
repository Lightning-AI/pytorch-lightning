import platform
from collections import namedtuple

import pytest
import torch
from packaging.version import parse as version_parse

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from tests.base import EvalModelTemplate


def test_early_stopping_cpu_model(tmpdir):
    """Test each of the trainer options."""
    stopping = EarlyStopping(monitor='val_loss', min_delta=0.1)
    trainer_options = dict(
        default_root_dir=tmpdir,
        early_stop_callback=stopping,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        train_percent_check=0.1,
        val_percent_check=0.1,
    )

    model = EvalModelTemplate()
    tutils.run_model_test(trainer_options, model, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


@pytest.mark.spawn
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif((platform.system() == "Darwin" and
                     version_parse(torch.__version__) < version_parse("1.3.0")),
                    reason="Distributed training is not supported on MacOS before Torch 1.3.0")
def test_multi_cpu_model_ddp(tmpdir):
    """Make sure DDP works."""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=None,
        num_processes=2,
        distributed_backend='ddp_cpu'
    )

    model = EvalModelTemplate()
    tutils.run_model_test(trainer_options, model, on_gpu=False)


def test_lbfgs_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=2,
        progress_bar_refresh_rate=0,
        weights_summary='top',
        train_percent_check=1.0,
        val_percent_check=0.2,
    )

    hparams = EvalModelTemplate.get_default_hparams()
    setattr(hparams, 'optimizer_name', 'lbfgs')
    setattr(hparams, 'learning_rate', 0.002)
    model = EvalModelTemplate(hparams)
    model.configure_optimizers = model.configure_optimizers__lbfgs
    tutils.run_model_test_without_loggers(trainer_options, model, min_acc=0.5)


def test_default_logger_callbacks_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        progress_bar_refresh_rate=0,
        train_percent_check=0.01,
        val_percent_check=0.01,
    )

    model = EvalModelTemplate()
    tutils.run_model_test_without_loggers(trainer_options, model)

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
        max_epochs=8,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
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
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
        early_stop_callback=False
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    tutils.assert_ok_model_acc(trainer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_single_gpu_batch_parse():
    trainer = Trainer()

    # batch is just a tensor
    batch = torch.rand(2, 3)
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch.device.index == 0 and batch.type() == 'torch.cuda.FloatTensor'

    # tensor list
    batch = [torch.rand(2, 3), torch.rand(2, 3)]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0].device.index == 0 and batch[0].type() == 'torch.cuda.FloatTensor'
    assert batch[1].device.index == 0 and batch[1].type() == 'torch.cuda.FloatTensor'

    # tensor list of lists
    batch = [[torch.rand(2, 3), torch.rand(2, 3)]]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'
    assert batch[0][1].device.index == 0 and batch[0][1].type() == 'torch.cuda.FloatTensor'

    # tensor dict
    batch = [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)}]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0]['a'].device.index == 0 and batch[0]['a'].type() == 'torch.cuda.FloatTensor'
    assert batch[0]['b'].device.index == 0 and batch[0]['b'].type() == 'torch.cuda.FloatTensor'

    # tuple of tensor list and list of tensor dict
    batch = ([torch.rand(2, 3) for _ in range(2)],
             [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)} for _ in range(2)])
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['a'].device.index == 0
    assert batch[1][0]['a'].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['b'].device.index == 0
    assert batch[1][0]['b'].type() == 'torch.cuda.FloatTensor'

    # namedtuple of tensor
    BatchType = namedtuple('BatchType', ['a', 'b'])
    batch = [BatchType(a=torch.rand(2, 3), b=torch.rand(2, 3)) for _ in range(2)]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0].a.device.index == 0
    assert batch[0].a.type() == 'torch.cuda.FloatTensor'


def test_simple_cpu(tmpdir):
    """Verify continue training session on CPU."""
    model = EvalModelTemplate()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.1,
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
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model = EvalModelTemplate()

    tutils.run_model_test(trainer_options, model, on_gpu=False)


def test_all_features_cpu_model(tmpdir):
    """Test each of the trainer options."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=2,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model = EvalModelTemplate()
    tutils.run_model_test(trainer_options, model, on_gpu=False)


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
        def __init__(self, hparams):
            super().__init__(hparams)
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

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    hparams = EvalModelTemplate.get_default_hparams()
    hparams.batch_size = batch_size
    hparams.in_features = truncated_bptt_steps
    hparams.hidden_dim = truncated_bptt_steps
    hparams.out_features = truncated_bptt_steps

    model = BpttTestModel(hparams)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        val_percent_check=0,
        weights_summary=None,
        early_stop_callback=False
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_single_gpu_model(tmpdir):
    """Make sure single GPU works (DP mode)."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus=1
    )

    model = EvalModelTemplate()
    tutils.run_model_test(trainer_options, model)
