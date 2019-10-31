import warnings

import pytest
import torch

from pytorch_lightning import Trainer, data_loader
from pytorch_lightning.callbacks import (
    EarlyStopping,
)
from pytorch_lightning.testing import (
    LightningTestModel,
    LightningTestModelBase,
    LightningTestMixin,
)
from . import testing_utils


def test_early_stopping_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    testing_utils.reset_seed()

    stopping = EarlyStopping(monitor='val_loss')
    trainer_options = dict(
        early_stop_callback=stopping,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=True,
        logger=testing_utils.get_test_tube_logger(),
        train_percent_check=0.1,
        val_percent_check=0.1
    )

    model, hparams = testing_utils.get_model()
    testing_utils.run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_lbfgs_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    testing_utils.reset_seed()

    trainer_options = dict(
        max_nb_epochs=1,
        print_nan_grads=True,
        show_progress_bar=False,
        weights_summary='top',
        train_percent_check=1.0,
        val_percent_check=0.2
    )

    model, hparams = testing_utils.get_model(use_test_model=True, lbfgs=True)
    testing_utils.run_model_test_no_loggers(trainer_options,
                                            model, hparams, on_gpu=False, min_acc=0.30)

    testing_utils.clear_save_dir()


def test_default_logger_callbacks_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    testing_utils.reset_seed()

    trainer_options = dict(
        max_nb_epochs=1,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        print_nan_grads=True,
        show_progress_bar=False,
        train_percent_check=0.01,
        val_percent_check=0.01
    )

    model, hparams = testing_utils.get_model()
    testing_utils.run_model_test_no_loggers(trainer_options, model, hparams, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()

    testing_utils.clear_save_dir()


def test_running_test_after_fitting():
    """Verify test() on fitted model"""
    testing_utils.reset_seed()

    hparams = testing_utils.get_hparams()
    model = LightningTestModel(hparams)

    save_dir = testing_utils.init_save_dir()

    # logger file to get meta
    logger = testing_utils.get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = testing_utils.init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    testing_utils.assert_ok_test_acc(trainer)

    testing_utils.clear_save_dir()


def test_running_test_without_val():
    testing_utils.reset_seed()

    """Verify test() works on a model with no val_loader"""

    class CurrentTestModel(LightningTestMixin, LightningTestModelBase):
        pass

    hparams = testing_utils.get_hparams()
    model = CurrentTestModel(hparams)

    save_dir = testing_utils.init_save_dir()

    # logger file to get meta
    logger = testing_utils.get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = testing_utils.init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    testing_utils.assert_ok_test_acc(trainer)

    testing_utils.clear_save_dir()


def test_single_gpu_batch_parse():
    testing_utils.reset_seed()

    if not testing_utils.can_run_gpu_test():
        return

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


def test_simple_cpu():
    """
    Verify continue training session on CPU
    :return:
    """
    testing_utils.reset_seed()

    hparams = testing_utils.get_hparams()
    model = LightningTestModel(hparams)

    save_dir = testing_utils.init_save_dir()

    # logger file to get meta
    trainer_options = dict(
        max_nb_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    testing_utils.clear_save_dir()


def test_cpu_model():
    """
    Make sure model trains on CPU
    :return:
    """
    testing_utils.reset_seed()

    trainer_options = dict(
        show_progress_bar=False,
        logger=testing_utils.get_test_tube_logger(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model, hparams = testing_utils.get_model()

    testing_utils.run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)


def test_all_features_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    testing_utils.reset_seed()

    trainer_options = dict(
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=False,
        logger=testing_utils.get_test_tube_logger(),
        accumulate_grad_batches=2,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model, hparams = testing_utils.get_model()
    testing_utils.run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)


def test_tbptt_cpu_model():
    """
    Test truncated back propagation through time works.
    :return:
    """
    testing_utils.reset_seed()

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

    class BpttTestModel(LightningTestModelBase):
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

            pred = self.forward(x_tensor.view(batch_size, truncated_bptt_steps))
            loss_val = torch.nn.functional.mse_loss(
                pred, y_tensor.view(batch_size, truncated_bptt_steps))
            return {
                'loss': loss_val,
                'hiddens': self.test_hidden,
            }

        @data_loader
        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    trainer_options = dict(
        max_nb_epochs=1,
        truncated_bptt_steps=truncated_bptt_steps,
        val_percent_check=0,
        weights_summary=None,
    )

    hparams = testing_utils.get_hparams()
    hparams.batch_size = batch_size
    hparams.in_features = truncated_bptt_steps
    hparams.hidden_dim = truncated_bptt_steps
    hparams.out_features = truncated_bptt_steps

    model = BpttTestModel(hparams)

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'


def test_single_gpu_model():
    """
    Make sure single GPU works (DP mode)
    :return:
    """
    testing_utils.reset_seed()

    if not torch.cuda.is_available():
        warnings.warn('test_single_gpu_model cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    model, hparams = testing_utils.get_model()

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus=1
    )

    testing_utils.run_gpu_model_test(trainer_options, model, hparams)


if __name__ == '__main__':
    pytest.main([__file__])
