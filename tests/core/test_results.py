import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult
import tests.base.develop_utils as tutils

from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule


def _setup_ddp(rank, worldsize):
    import os

    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _ddp_test_fn(rank, worldsize, result_cls: Result):
    _setup_ddp(rank, worldsize)
    tensor = torch.tensor([1.0])

    res = result_cls()
    res.log("test_tensor", tensor, sync_dist=True, sync_dist_op=torch.distributed.ReduceOp.SUM)

    assert res["test_tensor"].item() == dist.get_world_size(), "Result-Log does not work properly with DDP and Tensors"


@pytest.mark.parametrize("result_cls", [Result, TrainResult, EvalResult])
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_result_reduce_ddp(result_cls):
    """Make sure result logging works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, result_cls), nprocs=worldsize)


@pytest.mark.parametrize(
    "test_option,do_train,gpus",
    [
        pytest.param(
            0, True, 0, id='full_loop'
        ),
        pytest.param(
            0, False, 0, id='test_only'
        ),
        pytest.param(
            1, False, 0, id='test_only_mismatching_tensor', marks=pytest.mark.xfail(raises=ValueError, match="Mism.*")
        ),
        pytest.param(
            2, False, 0, id='mix_of_tensor_dims'
        ),
        pytest.param(
            3, False, 0, id='string_list_predictions'
        ),
        pytest.param(
            4, False, 0, id='int_list_predictions'
        ),
        pytest.param(
            5, False, 0, id='nested_list_predictions'
        ),
        pytest.param(
            6, False, 0, id='dict_list_predictions'
        ),
        pytest.param(
            0, True, 1, id='full_loop_single_gpu', marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires single-GPU machine")
        )
    ]
)
def test_result_obj_predictions(tmpdir, test_option, do_train, gpus):
    tutils.reset_seed()

    dm = TrialMNISTDataModule(tmpdir)
    prediction_file = Path('predictions.pt')

    model = EvalModelTemplate()
    model.test_option = test_option
    model.prediction_file = prediction_file.as_posix()
    model.test_step = model.test_step_result_preds
    model.test_step_end = None
    model.test_epoch_end = None
    model.test_end = None

    if prediction_file.exists():
        prediction_file.unlink()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
        gpus=gpus
    )

    # Prediction file shouldn't exist yet because we haven't done anything
    assert not prediction_file.exists()

    if do_train:
        result = trainer.fit(model, dm)
        assert result == 1
        result = trainer.test(datamodule=dm)
        result = result[0]
        assert result['test_loss'] < 0.6
        assert result['test_acc'] > 0.8
    else:
        result = trainer.test(model, datamodule=dm)

    # check prediction file now exists and is of expected length
    assert prediction_file.exists()
    predictions = torch.load(prediction_file)
    assert len(predictions) == len(dm.mnist_test)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_result_obj_predictions_ddp_spawn(tmpdir):
    distributed_backend = 'ddp_spawn'
    option = 0

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    seed_everything(4321)

    dm = TrialMNISTDataModule(tmpdir)

    prediction_file = Path('predictions.pt')

    model = EvalModelTemplate()
    model.test_option = option
    model.prediction_file = prediction_file.as_posix()
    model.test_step = model.test_step_result_preds
    model.test_step_end = None
    model.test_epoch_end = None
    model.test_end = None

    prediction_files = [Path('predictions_rank_0.pt'), Path('predictions_rank_1.pt')]
    for prediction_file in prediction_files:
        if prediction_file.exists():
            prediction_file.unlink()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
        distributed_backend=distributed_backend,
        gpus=[0, 1]
    )

    # Prediction file shouldn't exist yet because we haven't done anything
    # assert not model.prediction_file.exists()

    result = trainer.fit(model, dm)
    assert result == 1
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_loss'] < 0.6
    assert result['test_acc'] > 0.8

    dm.setup('test')

    # check prediction file now exists and is of expected length
    size = 0
    for prediction_file in prediction_files:
        assert prediction_file.exists()
        predictions = torch.load(prediction_file)
        size += len(predictions)
    assert size == len(dm.mnist_test)


def test_result_gather_stack():
    """ Test that tensors get concatenated when they all have the same shape. """
    outputs = [
        {"foo": torch.zeros(4, 5)},
        {"foo": torch.zeros(4, 5)},
        {"foo": torch.zeros(4, 5)},
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [12, 5]


def test_result_gather_concatenate():
    """ Test that tensors get concatenated when they have varying size in first dimension. """
    outputs = [
        {"foo": torch.zeros(4, 5)},
        {"foo": torch.zeros(8, 5)},
        {"foo": torch.zeros(3, 5)},
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [15, 5]


def test_result_gather_scalar():
    """ Test that 0-dim tensors get gathered and stacked correctly. """
    outputs = [
        {"foo": torch.tensor(1)},
        {"foo": torch.tensor(2)},
        {"foo": torch.tensor(3)},
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [3]


def test_result_gather_different_shapes():
    """ Test that tensors of varying shape get gathered into a list. """
    outputs = [
        {"foo": torch.tensor(1)},
        {"foo": torch.zeros(2, 3)},
        {"foo": torch.zeros(1, 2, 3)},
    ]
    result = Result.gather(outputs)
    expected = [torch.tensor(1), torch.zeros(2, 3), torch.zeros(1, 2, 3)]
    assert isinstance(result["foo"], list)
    assert all(torch.eq(r, e).all() for r, e in zip(result["foo"], expected))


def test_result_gather_mixed_types():
    """ Test that a collection of mixed types gets gathered into a list. """
    outputs = [
        {"foo": 1.2},
        {"foo": ["bar", None]},
        {"foo": torch.tensor(1)},
    ]
    result = Result.gather(outputs)
    expected = [1.2, ["bar", None], torch.tensor(1)]
    assert isinstance(result["foo"], list)
    assert result["foo"] == expected


def test_result_retrieve_last_logged_item():
    result = Result()
    result.log('a', 5., on_step=True, on_epoch=True)
    assert result['epoch_a'] == 5.
    assert result['step_a'] == 5.
    assert result['a'] == 5.
