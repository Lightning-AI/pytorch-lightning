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
from argparse import Namespace
from unittest import mock

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from packaging.version import Version
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_torch="1.5.0")
def test_tensorboard_hparams_reload(tmpdir):

    class CustomModel(BoringModel):

        def __init__(self, b1=0.5, b2=0.999):
            super().__init__()
            self.save_hyperparameters()

    trainer = Trainer(max_steps=1, default_root_dir=tmpdir)
    model = CustomModel()
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    assert trainer.log_dir == trainer.logger.log_dir
    folder_path = trainer.log_dir

    # make sure yaml is there
    with open(os.path.join(folder_path, "hparams.yaml")) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        yaml_params = yaml.safe_load(file)
        assert yaml_params["b1"] == 0.5
        assert yaml_params["b2"] == 0.999
        assert len(yaml_params.keys()) == 2

    # verify artifacts
    assert len(os.listdir(os.path.join(folder_path, "checkpoints"))) == 1

    # verify tb logs
    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    data_pt_1_5 = b'\x12\x1b"\x04\n\x02b1"\x04\n\x02b2*\r\n\x0b\x12\thp_metric'
    data_pt_1_6 = b'\x12\x1f"\x06\n\x02b1 \x03"\x06\n\x02b2 \x03*\r\n\x0b\x12\thp_metric'
    hparams_data = data_pt_1_6 if Version(torch.__version__) >= Version("1.6.0") else data_pt_1_5

    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.plugin_name == 'hparams'
    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.content == hparams_data


def test_tensorboard_automatic_versioning(tmpdir):
    """Verify that automatic versioning works"""

    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmpdir):
    """Verify that manual versioning works"""

    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning", version=1)

    assert logger.version == 1


def test_tensorboard_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402' """

    name = "tb_versioning"
    (tmpdir / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(save_dir=tmpdir, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmpdir / name) == [expected_version]
    assert os.listdir(tmpdir / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmpdir, name):
    """Verify that None or empty name works"""
    logger = TensorBoardLogger(save_dir=tmpdir, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert logger.root_dir == tmpdir
    assert os.listdir(tmpdir / "version_0")


def test_tensorboard_log_sub_dir(tmpdir):

    class TestLogger(TensorBoardLogger):
        # for reproducibility
        @property
        def version(self):
            return "version"

        @property
        def name(self):
            return "name"

    trainer_args = dict(
        default_root_dir=tmpdir,
        max_steps=1,
    )

    # no sub_dir specified
    save_dir = tmpdir / "logs"
    logger = TestLogger(save_dir)
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version")

    # sub_dir specified
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version", "sub_dir")

    # test home dir (`~`) handling
    save_dir = "~/tmp"
    explicit_save_dir = os.path.expanduser(save_dir)
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")

    # test env var (`$`) handling
    test_env_dir = "some_directory"
    os.environ["test_env_dir"] = test_env_dir
    save_dir = "$test_env_dir/tmp"
    explicit_save_dir = f"{test_env_dir}/tmp"
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmpdir, step_idx):
    logger = TensorBoardLogger(tmpdir)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1),
    }
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmpdir):
    logger = TensorBoardLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {
            "a": {
                "b": "c"
            }
        },
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
    }
    logger.log_hyperparams(hparams)


def test_tensorboard_log_hparams_and_metrics(tmpdir):
    logger = TensorBoardLogger(tmpdir, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {
            "a": {
                "b": "c"
            }
        },
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
    }
    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


def test_tensorboard_log_omegaconf_hparams_and_metrics(tmpdir):
    logger = TensorBoardLogger(tmpdir, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {
            "a": {
                "b": "c"
            }
        },
        "list": [1, 2, 3],
        # "namespace": Namespace(foo=Namespace(bar="buzz")),
        # "layer": torch.nn.BatchNorm1d,
    }
    hparams = OmegaConf.create(hparams)

    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_tensorboard_log_graph(tmpdir, example_input_array):
    """ test that log graph works with both model.example_input_array and
        if array is passed externaly
    """
    model = BoringModel()
    if example_input_array is not None:
        model.example_input_array = None

    logger = TensorBoardLogger(tmpdir, log_graph=True)
    logger.log_graph(model, example_input_array)


def test_tensorboard_log_graph_warning_no_example_input_array(tmpdir):
    """ test that log graph throws warning if model.example_input_array is None """
    model = BoringModel()
    model.example_input_array = None
    logger = TensorBoardLogger(tmpdir, log_graph=True)
    with pytest.warns(
        UserWarning,
        match='Could not log computational graph since the `model.example_input_array`'
        ' attribute is not set or `input_array` was not given'
    ):
        logger.log_graph(model)


@mock.patch('pytorch_lightning.loggers.TensorBoardLogger.log_metrics')
def test_tensorboard_with_accummulated_gradients(mock_log_metrics, tmpdir):
    """Tests to ensure that tensorboard log properly when accumulated_gradients > 1"""

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.indexes = []

        def training_step(self, *args):
            self.log('foo', 1, on_step=True, on_epoch=True)
            if not self.trainer.train_loop.should_accumulate():
                if self.trainer.logger_connector.should_update_logs:
                    self.indexes.append(self.trainer.global_step)
            return super().training_step(*args)

    model = TestModel()
    model.training_epoch_end = None
    logger_0 = TensorBoardLogger(tmpdir, default_hp_metric=False)
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=12,
        limit_val_batches=0,
        max_epochs=3,
        accumulate_grad_batches=2,
        logger=[logger_0],
        log_every_n_steps=3,
    )
    trainer.fit(model)

    calls = [m[2] for m in mock_log_metrics.mock_calls]
    count_epochs = [c["step"] for c in calls if "foo_epoch" in c["metrics"]]
    assert count_epochs == [5, 11, 17]

    count_steps = [c["step"] for c in calls if "foo_step" in c["metrics"]]
    assert count_steps == model.indexes


@mock.patch('pytorch_lightning.loggers.tensorboard.SummaryWriter')
def test_tensorboard_finalize(summary_writer, tmpdir):
    """ Test that the SummaryWriter closes in finalize. """
    logger = TensorBoardLogger(save_dir=tmpdir)
    logger.finalize("any")
    summary_writer().flush.assert_called()
    summary_writer().close.assert_called()


def test_tensorboard_save_hparams_to_yaml_once(tmpdir):
    model = BoringModel()
    logger = TensorBoardLogger(save_dir=tmpdir, default_hp_metric=False)
    trainer = Trainer(max_steps=1, default_root_dir=tmpdir, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    hparams_file = "hparams.yaml"
    assert os.path.isfile(os.path.join(trainer.log_dir, hparams_file))
    assert not os.path.isfile(os.path.join(tmpdir, hparams_file))


@mock.patch('pytorch_lightning.loggers.tensorboard.log')
def test_tensorboard_with_symlink(log, tmpdir):
    """
    Tests a specific failure case when tensorboard logger is used with empty name, symbolic link ``save_dir``, and
    relative paths.
    """
    os.chdir(tmpdir)  # need to use relative paths
    source = os.path.join('.', 'lightning_logs')
    dest = os.path.join('.', 'sym_lightning_logs')

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = TensorBoardLogger(save_dir=dest, name='')
    _ = logger.version

    log.warning.assert_not_called()
