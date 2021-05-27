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
from argparse import ArgumentParser
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.plugins import TPUSpawnPlugin
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test

if _TPU_AVAILABLE:
    import torch_xla
    import torch_xla.distributed.xla_multiprocessing as xmp
    SERIAL_EXEC = xmp.MpSerialExecutor()

_LARGER_DATASET = RandomDataset(32, 2000)


# 8 cores needs a big dataset
def _serial_train_loader():
    return DataLoader(_LARGER_DATASET, batch_size=32)


class SerialLoaderBoringModel(BoringModel):

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        tpu_cores=1,
        limit_train_batches=4,
        limit_val_batches=4,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.parametrize('tpu_core', [1, 5])
@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        tpu_cores=[tpu_core],
        limit_train_batches=4,
        limit_val_batches=4,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)
    assert torch_xla._XLAC._xla_get_default_device() == f'xla:{tpu_core}'


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=4,
        limit_val_batches=4,
    )

    # 8 cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False, min_acc=0.05)


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_16bit_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        tpu_cores=1,
        limit_train_batches=8,
        limit_val_batches=2,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.parametrize('tpu_core', [1, 5])
@RunIf(tpu=True)
@pl_multi_process_test
def test_model_16bit_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        tpu_cores=[tpu_core],
        limit_train_batches=4,
        limit_val_batches=2,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert torch_xla._XLAC._xla_get_default_device() == f'xla:{tpu_core}'
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_16bit_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=4,
        limit_val_batches=4,
    )

    # 8 cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False, min_acc=0.05)


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_early_stop(tmpdir):
    """Test if single TPU core training works"""

    class CustomBoringModel(BoringModel):

        def validation_step(self, *args, **kwargs):
            out = super().validation_step(*args, **kwargs)
            self.log('val_loss', out['x'])
            return out

    tutils.reset_seed()
    model = CustomBoringModel()
    trainer = Trainer(
        callbacks=[EarlyStopping(monitor='val_loss')],
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        tpu_cores=8,
    )
    trainer.fit(model)
    trainer.test(test_dataloaders=DataLoader(RandomDataset(32, 2000), batch_size=32))


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_grad_norm(tmpdir):
    """Test if grad_norm works on TPU."""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=4,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
        gradient_clip_val=0.5,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_clip_grad_by_value(tmpdir):
    """Test if clip_gradients by value works on TPU"""
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=4,
        tpu_cores=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value'
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@RunIf(tpu=True)
@pl_multi_process_test
def test_dataloaders_passed_to_fit(tmpdir):
    """Test if dataloaders passed to trainer works on TPU"""
    tutils.reset_seed()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        tpu_cores=8,
    )
    trainer.fit(
        model,
        train_dataloader=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
    )
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    ['tpu_cores', 'expected_tpu_id'],
    [pytest.param(1, None), pytest.param(8, None),
     pytest.param([1], 1), pytest.param([8], 8)],
)
@RunIf(tpu=True)
def test_tpu_id_to_be_as_expected(tpu_cores, expected_tpu_id):
    """Test if trainer.tpu_id is set as expected"""
    assert Trainer(tpu_cores=tpu_cores).accelerator_connector.tpu_id == expected_tpu_id


def test_tpu_misconfiguration():
    """Test if trainer.tpu_id is set as expected"""
    with pytest.raises(MisconfigurationException, match="`tpu_cores` can only be"):
        Trainer(tpu_cores=[1, 8])


@pytest.mark.skipif(_TPU_AVAILABLE, reason="test requires missing TPU")
def test_exception_when_no_tpu_found(tmpdir):
    """Test if exception is thrown when xla devices are not available"""

    with pytest.raises(MisconfigurationException, match='No TPU devices were found.'):
        Trainer(tpu_cores=8)


@pytest.mark.parametrize('tpu_cores', [1, 8, [1]])
@RunIf(tpu=True)
def test_distributed_backend_set_when_using_tpu(tmpdir, tpu_cores):
    """Test if distributed_backend is set to `tpu` when tpu_cores is not None"""
    assert Trainer(tpu_cores=tpu_cores).distributed_backend == "tpu"


@RunIf(tpu=True)
@pl_multi_process_test
def test_broadcast_on_tpu():
    """ Checks if an object from the master process is broadcasted to other processes correctly"""

    def test_broadcast(rank):
        trainer = Trainer(tpu_cores=8)
        assert isinstance(trainer.accelerator, TPUAccelerator)
        assert isinstance(trainer.training_type_plugin, TPUSpawnPlugin)
        obj = ("ver_0.5", "logger_name", rank)
        result = trainer.training_type_plugin.broadcast(obj)
        assert result == ("ver_0.5", "logger_name", 0)

    xmp.spawn(test_broadcast, nprocs=8, start_method='fork')


@pytest.mark.parametrize(
    ["tpu_cores", "expected_tpu_id", "error_expected"],
    [
        pytest.param(1, None, False),
        pytest.param(8, None, False),
        pytest.param([1], 1, False),
        pytest.param([8], 8, False),
        pytest.param("1,", 1, False),
        pytest.param("1", None, False),
        pytest.param("9, ", 9, True),
        pytest.param([9], 9, True),
        pytest.param([0], 0, True),
        pytest.param(2, None, True),
        pytest.param(10, None, True),
    ],
)
@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_choice(tmpdir, tpu_cores, expected_tpu_id, error_expected):
    if error_expected:
        with pytest.raises(MisconfigurationException, match=r".*tpu_cores` can only be 1, 8 or [<1-8>]*"):
            Trainer(default_root_dir=tmpdir, tpu_cores=tpu_cores)
    else:
        trainer = Trainer(default_root_dir=tmpdir, tpu_cores=tpu_cores)
        assert trainer.accelerator_connector.tpu_id == expected_tpu_id


@pytest.mark.parametrize(
    ['cli_args', 'expected'],
    [pytest.param('--tpu_cores=8', {'tpu_cores': 8}),
     pytest.param("--tpu_cores=1,", {'tpu_cores': '1,'})]
)
@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_cores_with_argparse(cli_args, expected):
    """Test passing tpu_cores in command line"""
    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    for k, v in expected.items():
        assert getattr(args, k) == v
    assert Trainer.from_argparse_args(args)


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_reduce():
    """Test tpu spawn reduce operation """

    def test_reduce(rank):
        trainer = Trainer(tpu_cores=8)
        # faster this way
        reduce_ops = ["mean", "AVG", "undefined", "sum", ReduceOp.SUM, ReduceOp.MAX]
        for reduce_op in reduce_ops:
            if reduce_op == "undefined" or reduce_op == ReduceOp.MAX:
                with pytest.raises(MisconfigurationException, match="TPUSpawn TrainingTypePlugin only support"):
                    result = trainer.training_type_plugin.reduce(1, reduce_op)
            else:
                result = trainer.training_type_plugin.reduce(1, reduce_op)
            if isinstance(reduce_op, str) and reduce_op.lower() in ("mean", "avg"):
                assert result.item() == 1
            else:
                assert result.item() == 8

    xmp.spawn(test_reduce, nprocs=8, start_method='fork')


@RunIf(tpu=True)
@pl_multi_process_test
@pytest.mark.parametrize("clip_val", [10])
@mock.patch("torch.nn.utils.clip_grad_norm_")
def test_tpu_precision_16_clip_gradients(mock_clip_grad_norm, clip_val, tmpdir):
    """
    Ensure that clip gradients is only called if the value is greater than 0.
    TODO: Fix (test fails with parametrize)
    """
    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        precision=16,
        limit_train_batches=4,
        limit_val_batches=4,
        gradient_clip_val=clip_val,
    )
    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)

    if clip_val > 0:
        mock_clip_grad_norm.assert_called()
    else:
        mock_clip_grad_norm.assert_not_called()


@RunIf(tpu=True)
@pl_multi_process_test
def test_if_test_works_with_checkpoint_false(tmpdir):
    """Ensure that model trains properly when `checkpoint_callback` is set to False."""

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, tpu_cores=8, default_root_dir=tmpdir, fast_dev_run=True, checkpoint_callback=False)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_sync_dist():
    """Test tpu spawn sync dist operation """

    def test_sync_dist(rank):
        tensor = torch.tensor([1.0])
        training_type_plugin = TPUSpawnPlugin()

        res = Result()
        res.log(
            "test_tensor",
            tensor,
            sync_fn=training_type_plugin.reduce,
            sync_dist=True,
            sync_dist_op=torch.distributed.ReduceOp.SUM
        )

        assert res["test_tensor"].item() == 8, "Result-Log does not work properly with TPU Spawn and Tensors"

    xmp.spawn(test_sync_dist, nprocs=8, start_method='fork')


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_debug_mode(tmpdir):
    """Test if debug mode works on TPU."""

    class DebugModel(BoringModel):

        def on_train_start(self):
            assert os.environ.get("PT_XLA_DEBUG") == str(1), "PT_XLA_DEBUG was not set in environment variables"

        def teardown(self, stage):
            assert "PT_XLA_DEBUG" not in os.environ

    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=4,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
        plugins=TPUSpawnPlugin(debug=True),
    )

    model = DebugModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@RunIf(tpu=True)
@pl_multi_process_test
def test_tpu_host_world_size(tmpdir):
    """Test Host World size env setup on TPU."""

    class DebugModel(BoringModel):

        def on_train_start(self):
            assert os.environ.get("XRT_HOST_WORLD_SIZE") == str(1)

        def teardown(self, stage):
            assert "XRT_HOST_WORLD_SIZE" not in os.environ

    tutils.reset_seed()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=4,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = DebugModel()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)
