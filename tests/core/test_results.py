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
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.states import TrainerState
from tests import _SKIPIF_ARGS_NO_GPU
from tests.helpers import BoringDataModule, BoringModel


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


@pytest.mark.parametrize("result_cls", [Result])
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_result_reduce_ddp(result_cls):
    """Make sure result logging works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, result_cls), nprocs=worldsize)


@pytest.mark.parametrize(
    "test_option,do_train,gpus", [
        pytest.param(0, True, 0, id='full_loop'),
        pytest.param(0, False, 0, id='test_only'),
        pytest.param(
            1, False, 0, id='test_only_mismatching_tensor', marks=pytest.mark.xfail(raises=ValueError, match="Mism.*")
        ),
        pytest.param(2, False, 0, id='mix_of_tensor_dims'),
        pytest.param(3, False, 0, id='string_list_predictions'),
        pytest.param(4, False, 0, id='int_list_predictions'),
        pytest.param(5, False, 0, id='nested_list_predictions'),
        pytest.param(6, False, 0, id='dict_list_predictions'),
        pytest.param(7, True, 0, id='write_dict_predictions'),
        pytest.param(0, True, 1, id='full_loop_single_gpu', marks=pytest.mark.skipif(**_SKIPIF_ARGS_NO_GPU))
    ]
)
def test_result_obj_predictions(tmpdir, test_option, do_train, gpus):

    class CustomBoringModel(BoringModel):

        def test_step(self, batch, batch_idx, optimizer_idx=None):
            output = self(batch)
            test_loss = self.loss(batch, output)
            self.log('test_loss', test_loss)

            batch_size = batch.size(0)
            lst_of_str = [random.choice(['dog', 'cat']) for i in range(batch_size)]
            lst_of_int = [random.randint(500, 1000) for i in range(batch_size)]
            lst_of_lst = [[x] for x in lst_of_int]
            lst_of_dict = [{k: v} for k, v in zip(lst_of_str, lst_of_int)]

            # This is passed in from pytest via parameterization
            option = getattr(self, 'test_option', 0)
            prediction_file = getattr(self, 'prediction_file', 'predictions.pt')

            lazy_ids = torch.arange(batch_idx * batch_size, batch_idx * batch_size + batch_size)

            # Base
            if option == 0:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('preds', output, prediction_file)

            # Check mismatching tensor len
            elif option == 1:
                self.write_prediction('idxs', torch.cat((lazy_ids, lazy_ids)), prediction_file)
                self.write_prediction('preds', output, prediction_file)

            # write multi-dimension
            elif option == 2:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('preds', output, prediction_file)
                self.write_prediction('x', batch, prediction_file)

            # write str list
            elif option == 3:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('vals', lst_of_str, prediction_file)

            # write int list
            elif option == 4:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('vals', lst_of_int, prediction_file)

            # write nested list
            elif option == 5:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('vals', lst_of_lst, prediction_file)

            # write dict list
            elif option == 6:
                self.write_prediction('idxs', lazy_ids, prediction_file)
                self.write_prediction('vals', lst_of_dict, prediction_file)

            elif option == 7:
                self.write_prediction_dict({'idxs': lazy_ids, 'preds': output}, prediction_file)

    class CustomBoringDataModule(BoringDataModule):

        def train_dataloader(self):
            return DataLoader(self.random_train, batch_size=4)

        def val_dataloader(self):
            return DataLoader(self.random_val, batch_size=4)

        def test_dataloader(self):
            return DataLoader(self.random_test, batch_size=4)

    tutils.reset_seed()
    prediction_file = Path(tmpdir) / 'predictions.pt'

    dm = BoringDataModule()
    model = CustomBoringModel()
    model.test_step_end = None
    model.test_epoch_end = None
    model.test_end = None

    model.test_option = test_option
    model.prediction_file = prediction_file.as_posix()

    if prediction_file.exists():
        prediction_file.unlink()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
        gpus=gpus,
    )

    # Prediction file shouldn't exist yet because we haven't done anything
    assert not prediction_file.exists()

    if do_train:
        result = trainer.fit(model, dm)
        assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
        assert result
        result = trainer.test(datamodule=dm)
        # TODO: add end-to-end test
        # assert result[0]['test_loss'] < 0.6
    else:
        result = trainer.test(model, datamodule=dm)

    # check prediction file now exists and is of expected length
    assert prediction_file.exists()
    predictions = torch.load(prediction_file)
    assert len(predictions) == len(dm.random_test)


def test_result_gather_stack():
    """ Test that tensors get concatenated when they all have the same shape. """
    outputs = [
        {
            "foo": torch.zeros(4, 5)
        },
        {
            "foo": torch.zeros(4, 5)
        },
        {
            "foo": torch.zeros(4, 5)
        },
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [12, 5]


def test_result_gather_concatenate():
    """ Test that tensors get concatenated when they have varying size in first dimension. """
    outputs = [
        {
            "foo": torch.zeros(4, 5)
        },
        {
            "foo": torch.zeros(8, 5)
        },
        {
            "foo": torch.zeros(3, 5)
        },
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [15, 5]


def test_result_gather_scalar():
    """ Test that 0-dim tensors get gathered and stacked correctly. """
    outputs = [
        {
            "foo": torch.tensor(1)
        },
        {
            "foo": torch.tensor(2)
        },
        {
            "foo": torch.tensor(3)
        },
    ]
    result = Result.gather(outputs)
    assert isinstance(result["foo"], torch.Tensor)
    assert list(result["foo"].shape) == [3]


def test_result_gather_different_shapes():
    """ Test that tensors of varying shape get gathered into a list. """
    outputs = [
        {
            "foo": torch.tensor(1)
        },
        {
            "foo": torch.zeros(2, 3)
        },
        {
            "foo": torch.zeros(1, 2, 3)
        },
    ]
    result = Result.gather(outputs)
    expected = [torch.tensor(1), torch.zeros(2, 3), torch.zeros(1, 2, 3)]
    assert isinstance(result["foo"], list)
    assert all(torch.eq(r, e).all() for r, e in zip(result["foo"], expected))


def test_result_gather_mixed_types():
    """ Test that a collection of mixed types gets gathered into a list. """
    outputs = [
        {
            "foo": 1.2
        },
        {
            "foo": ["bar", None]
        },
        {
            "foo": torch.tensor(1)
        },
    ]
    result = Result.gather(outputs)
    expected = [1.2, ["bar", None], torch.tensor(1)]
    assert isinstance(result["foo"], list)
    assert result["foo"] == expected


def test_result_retrieve_last_logged_item():
    result = Result()
    result.log('a', 5., on_step=True, on_epoch=True)
    assert result['a_epoch'] == 5.
    assert result['a_step'] == 5.
    assert result['a'] == 5.
