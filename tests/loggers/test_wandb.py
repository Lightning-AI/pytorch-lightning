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
import pickle
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.utils import no_warning_call


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_logger_init(wandb, monkeypatch):
    """Verify that basic functionality of wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.
    """
    import pytorch_lightning.loggers.wandb as imports

    # test wandb.init called when there is no W&B run
    wandb.run = None
    logger = WandbLogger(
        name="test_name", save_dir="test_save_dir", version="test_id", project="test_project", resume="never"
    )
    logger.log_metrics({"acc": 1.0})
    wandb.init.assert_called_once_with(
        name="test_name", dir="test_save_dir", id="test_id", project="test_project", resume="never", anonymous=None
    )
    wandb.init().log.assert_called_once_with({"acc": 1.0})

    # test wandb.init and setting logger experiment externally
    wandb.run = None
    run = wandb.init()
    logger = WandbLogger(experiment=run)
    assert logger.experiment

    # test wandb.init not called if there is a W&B run
    wandb.init().log.reset_mock()
    wandb.init.reset_mock()
    wandb.run = wandb.init()

    monkeypatch.setattr(imports, "_WANDB_GREATER_EQUAL_0_12_10", True)
    with pytest.warns(UserWarning, match="There is a wandb run already in progress"):
        logger = WandbLogger()
    # check that no new run is created
    with no_warning_call(UserWarning, match="There is a wandb run already in progress"):
        _ = logger.experiment

    # verify default resume value
    assert logger._wandb_init["resume"] == "allow"

    logger.log_metrics({"acc": 1.0}, step=3)
    wandb.init.assert_called_once()
    wandb.init().log.assert_called_once_with({"acc": 1.0, "trainer/global_step": 3})

    # continue training on same W&B run and offset step
    logger.finalize("success")
    logger.log_metrics({"acc": 1.0}, step=6)
    wandb.init().log.assert_called_with({"acc": 1.0, "trainer/global_step": 6})

    # log hyper parameters
    logger.log_hyperparams({"test": None, "nested": {"a": 1}, "b": [2, 3, 4]})
    wandb.init().config.update.assert_called_once_with(
        {"test": "None", "nested/a": 1, "b": [2, 3, 4]}, allow_val_change=True
    )

    # watch a model
    logger.watch("model", "log", 10, False)
    wandb.init().watch.assert_called_once_with("model", log="log", log_freq=10, log_graph=False)

    assert logger.name == wandb.init().project_name()
    assert logger.version == wandb.init().id


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_pickle(wandb, tmpdir):
    """Verify that pickling trainer with wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.
    """

    class Experiment:
        id = "the_id"
        step = 0
        dir = "wandb"

        def project_name(self):
            return "the_project_name"

    wandb.run = None
    wandb.init.return_value = Experiment()
    logger = WandbLogger(id="the_id", offline=True)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, logger=logger)
    # Access the experiment to ensure it's created
    assert trainer.logger.experiment, "missing experiment"
    assert trainer.log_dir == logger.save_dir
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)

    assert os.environ["WANDB_MODE"] == "dryrun"
    assert trainer2.logger.__class__.__name__ == WandbLogger.__name__
    assert trainer2.logger.experiment, "missing experiment"

    wandb.init.assert_called()
    assert "id" in wandb.init.call_args[1]
    assert wandb.init.call_args[1]["id"] == "the_id"

    del os.environ["WANDB_MODE"]


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_logger_dirs_creation(wandb, monkeypatch, tmpdir):
    """Test that the logger creates the folders and files in the right place."""
    import pytorch_lightning.loggers.wandb as imports

    monkeypatch.setattr(imports, "_WANDB_GREATER_EQUAL_0_12_10", True)
    wandb.run = None
    logger = WandbLogger(save_dir=str(tmpdir), offline=True)
    # the logger get initialized
    assert logger.version == wandb.init().id
    assert logger.name == wandb.init().project_name()

    # mock return values of experiment
    wandb.run = None
    logger.experiment.id = "1"
    logger.experiment.project_name.return_value = "project"

    for _ in range(2):
        _ = logger.experiment

    assert logger.version == "1"
    assert logger.name == "project"
    assert str(tmpdir) == logger.save_dir
    assert not os.listdir(tmpdir)

    version = logger.version
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_train_batches=3, limit_val_batches=3)
    assert trainer.log_dir == logger.save_dir
    trainer.fit(model)

    assert trainer.checkpoint_callback.dirpath == str(tmpdir / "project" / version / "checkpoints")
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {"epoch=0-step=3.ckpt"}
    assert trainer.log_dir == logger.save_dir


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_log_model(wandb, monkeypatch, tmpdir):
    """Test that the logger creates the folders and files in the right place."""
    import pytorch_lightning.loggers.wandb as imports

    monkeypatch.setattr(imports, "_WANDB_GREATER_EQUAL_0_10_22", True)

    wandb.run = None
    model = BoringModel()

    # test log_model=True
    logger = WandbLogger(log_model=True)
    logger.experiment.id = "1"
    logger.experiment.project_name.return_value = "project"
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3)
    trainer.fit(model)
    wandb.init().log_artifact.assert_called_once()

    # test log_model='all'
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    logger = WandbLogger(log_model="all")
    logger.experiment.id = "1"
    logger.experiment.project_name.return_value = "project"
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3)
    trainer.fit(model)
    assert wandb.init().log_artifact.call_count == 2

    # test log_model=False
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    logger = WandbLogger(log_model=False)
    logger.experiment.id = "1"
    logger.experiment.project_name.return_value = "project"
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3)
    trainer.fit(model)
    assert not wandb.init().log_artifact.called

    # test correct metadata
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(log_model=True)
    logger.experiment.id = "1"
    logger.experiment.project_name.return_value = "project"
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3)
    trainer.fit(model)
    wandb.Artifact.assert_called_once_with(
        name="model-1",
        type="model",
        metadata={
            "score": None,
            "original_filename": "epoch=1-step=6-v3.ckpt",
            "ModelCheckpoint": {
                "monitor": None,
                "mode": "min",
                "save_last": None,
                "save_top_k": 1,
                "save_weights_only": False,
                "_every_n_train_steps": 0,
            },
        },
    )


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_log_media(wandb, tmpdir):
    """Test that the logger creates the folders and files in the right place."""

    wandb.run = None

    # test log_text with columns and data
    columns = ["input", "label", "prediction"]
    data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]
    logger = WandbLogger()
    logger.log_text(key="samples", columns=columns, data=data)
    wandb.Table.assert_called_once_with(
        columns=["input", "label", "prediction"],
        data=[["cheese", "english", "english"], ["fromage", "french", "spanish"]],
        dataframe=None,
    )
    wandb.init().log.assert_called_once_with({"samples": wandb.Table()})

    # test log_text with dataframe
    wandb.Table.reset_mock()
    wandb.init().log.reset_mock()
    df = 'pandas.DataFrame({"col1": [1, 2], "col2": [3, 4]})'  # TODO: incompatible numpy/pandas versions in test env
    logger.log_text(key="samples", dataframe=df)
    wandb.Table.assert_called_once_with(
        columns=None,
        data=None,
        dataframe=df,
    )
    wandb.init().log.assert_called_once_with({"samples": wandb.Table()})

    # test log_image
    wandb.init().log.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"])
    wandb.Image.assert_called_with("2.jpg")
    wandb.init().log.assert_called_once_with({"samples": [wandb.Image(), wandb.Image()]})

    # test log_image with step
    wandb.init().log.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"], step=5)
    wandb.Image.assert_called_with("2.jpg")
    wandb.init().log.assert_called_once_with({"samples": [wandb.Image(), wandb.Image()], "trainer/global_step": 5})

    # test log_image with captions
    wandb.init().log.reset_mock()
    wandb.Image.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"], caption=["caption 1", "caption 2"])
    wandb.Image.assert_called_with("2.jpg", caption="caption 2")
    wandb.init().log.assert_called_once_with({"samples": [wandb.Image(), wandb.Image()]})

    # test log_image without a list
    with pytest.raises(TypeError, match="""Expected a list as "images", found <class 'str'>"""):
        logger.log_image(key="samples", images="1.jpg")

    # test log_image with wrong number of captions
    with pytest.raises(ValueError, match="Expected 2 items but only found 1 for caption"):
        logger.log_image(key="samples", images=["1.jpg", "2.jpg"], caption=["caption 1"])

    # test log_table
    wandb.Table.reset_mock()
    wandb.init().log.reset_mock()
    logger.log_table(key="samples", columns=columns, data=data, dataframe=df, step=5)
    wandb.Table.assert_called_once_with(
        columns=columns,
        data=data,
        dataframe=df,
    )
    wandb.init().log.assert_called_once_with({"samples": wandb.Table(), "trainer/global_step": 5})


@mock.patch("pytorch_lightning.loggers.wandb.wandb")
def test_wandb_logger_offline_log_model(wandb, tmpdir):
    """Test that log_model=True raises an error in offline mode."""
    with pytest.raises(MisconfigurationException, match="checkpoints cannot be uploaded in offline mode"):
        _ = WandbLogger(save_dir=str(tmpdir), offline=True, log_model=True)
