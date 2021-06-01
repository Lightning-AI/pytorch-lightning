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
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tests.helpers.utils as tutils
from pytorch_lightning import LightningModule, Trainer
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.runif import RunIf


def _setup_ddp(rank, worldsize):
    import os

    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _ddp_test_fn(rank, worldsize):
    _setup_ddp(rank, worldsize)
    tensor = torch.tensor([1.0])
    actual = LightningModule._LightningModule__sync(tensor, sync_dist=True, sync_dist_op=torch.distributed.ReduceOp.SUM)
    assert actual.item() == dist.get_world_size(), "Result-Log does not work properly with DDP and Tensors"


@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, ), nprocs=worldsize)


@pytest.mark.parametrize(["option", "do_train", "gpus"], [
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
    pytest.param(0, True, 1, id='full_loop_single_gpu', marks=RunIf(min_gpus=1))
])
def test_write_predictions(tmpdir, option: int, do_train: bool, gpus: int):

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

    prediction_file = Path(tmpdir) / 'predictions.pt'

    dm = BoringDataModule()
    model = CustomBoringModel()
    model.test_epoch_end = None
    model.prediction_file = prediction_file.as_posix()

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
        trainer.fit(model, dm)
        assert trainer.state.finished, f"Training failed with {trainer.state}"
        trainer.test(datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)

    # check prediction file now exists and is of expected length
    assert prediction_file.exists()
    predictions = torch.load(prediction_file)
    assert len(predictions) == len(dm.random_test)
