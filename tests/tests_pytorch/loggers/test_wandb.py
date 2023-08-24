# Copyright The Lightning AI team.
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
from lightning_utilities.test.warning import no_warning_call

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_project_name(*_):
    with mock.patch.dict(os.environ, {}):
        logger = WandbLogger()
    assert logger.name == "lightning_logs"

    with mock.patch.dict(os.environ, {}):
        logger = WandbLogger(project="project")
    assert logger.name == "project"

    with mock.patch.dict(os.environ, {"WANDB_PROJECT": "env_project"}):
        logger = WandbLogger()
    assert logger.name == "env_project"

    with mock.patch.dict(os.environ, {"WANDB_PROJECT": "env_project"}):
        logger = WandbLogger(project="project")
    assert logger.name == "project"


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_logger_init(wandb, monkeypatch):
    """Verify that basic functionality of wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.

    """
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

    # test wandb.init called with project as name if name not provided
    wandb.run = None
    wandb.init.reset_mock()
    WandbLogger(project="test_project").experiment
    wandb.init.assert_called_once_with(
        name=None, dir=".", id=None, project="test_project", resume="allow", anonymous=None
    )

    # test wandb.init set save_dir correctly after created
    wandb.run = None
    wandb.init.reset_mock()
    logger = WandbLogger()
    assert logger.save_dir is not None
    wandb.run = None
    wandb.init.reset_mock()
    logger = WandbLogger(save_dir=".", dir=None)
    assert logger.save_dir is not None

    # test wandb.init and setting logger experiment externally
    wandb.run = None
    run = wandb.init()
    logger = WandbLogger(experiment=run)
    assert logger.experiment

    # test wandb.init not called if there is a W&B run
    wandb.init().log.reset_mock()
    wandb.init.reset_mock()
    wandb.run = wandb.init()

    logger = WandbLogger()
    with pytest.warns(UserWarning, match="There is a wandb run already in progress"):
        _ = logger.experiment

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
    hparams = {"test": None, "nested": {"a": 1}, "b": [2, 3, 4]}
    logger.log_hyperparams(hparams)
    wandb.init().config.update.assert_called_once_with(hparams, allow_val_change=True)

    # watch a model
    logger.watch("model", "log", 10, False)
    wandb.init().watch.assert_called_once_with("model", log="log", log_freq=10, log_graph=False)

    assert logger.version == wandb.init().id


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_logger_init_before_spawn(_, monkeypatch):
    logger = WandbLogger()
    assert logger._experiment is None
    logger.__getstate__()
    assert logger._experiment is not None


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_pickle(wandb, tmp_path):
    """Verify that pickling trainer with wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.

    """

    class Experiment:
        id = "the_id"
        step = 0
        dir = "wandb"

        @property
        def name(self):
            return "the_run_name"

    with mock.patch("lightning.pytorch.loggers.wandb.Run", new=Experiment):
        wandb.run = None
        wandb.init.return_value = Experiment()
        logger = WandbLogger(id="the_id", offline=True)

        trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, logger=logger)
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


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_logger_dirs_creation(wandb, monkeypatch, tmp_path):
    """Test that the logger creates the folders and files in the right place."""
    wandb.run = None
    logger = WandbLogger(project="project", save_dir=str(tmp_path), offline=True)

    # mock return values of experiment
    wandb.run = None
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"

    for _ in range(2):
        _ = logger.experiment

    assert logger.version == "1"
    assert logger.name == "project"
    assert str(tmp_path) == logger.save_dir
    assert not os.listdir(tmp_path)

    version = logger.version
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=1, limit_train_batches=3, limit_val_batches=3
    )
    assert trainer.log_dir == logger.save_dir
    trainer.fit(model)

    assert trainer.checkpoint_callback.dirpath == str(tmp_path / "project" / version / "checkpoints")
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {"epoch=0-step=3.ckpt"}
    assert trainer.log_dir == logger.save_dir


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_log_model(wandb, monkeypatch, tmp_path):
    """Test that the logger creates the folders and files in the right place."""
    wandb.run = None
    model = BoringModel()

    # test log_model=True
    logger = WandbLogger(save_dir=tmp_path, log_model=True)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3
    )
    trainer.fit(model)
    wandb.init().log_artifact.assert_called_once()

    # test log_model='all'
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model="all")
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3
    )
    trainer.fit(model)
    assert wandb.init().log_artifact.call_count == 2

    # test log_model=False
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=False)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3
    )
    trainer.fit(model)
    assert not wandb.init().log_artifact.called

    # test correct metadata
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=True)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3
    )
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

    # Test wandb custom artifact name
    wandb.init().log_artifact.reset_mock()
    wandb.init().reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=True, checkpoint_name="my-test-model")
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=2, limit_train_batches=3, limit_val_batches=3
    )
    trainer.fit(model)
    wandb.Artifact.assert_called_once_with(
        name="my-test-model",
        type="model",
        metadata={
            "score": None,
            "original_filename": "epoch=1-step=6-v4.ckpt",
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

    # Test wandb artifact with checkpoint_callback top_k logging latest
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=True)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=3,
        callbacks=[ModelCheckpoint(monitor="step", save_top_k=2)],
    )
    trainer.fit(model)
    wandb.Artifact.assert_called_with(
        name="model-1",
        type="model",
        metadata={
            "score": 6,
            "original_filename": "epoch=1-step=6-v5.ckpt",
            "ModelCheckpoint": {
                "monitor": "step",
                "mode": "min",
                "save_last": None,
                "save_top_k": 2,
                "save_weights_only": False,
                "_every_n_train_steps": 0,
            },
        },
    )
    wandb.init().log_artifact.assert_called_with(wandb.Artifact(), aliases=["latest"])

    # Test wandb artifact with checkpoint_callback top_k logging latest and best
    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=True)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=3,
        callbacks=[
            ModelCheckpoint(
                monitor="step",
            )
        ],
    )
    trainer.fit(model)
    wandb.Artifact.assert_called_with(
        name="model-1",
        type="model",
        metadata={
            "score": 3,
            "original_filename": "epoch=0-step=3-v1.ckpt",
            "ModelCheckpoint": {
                "monitor": "step",
                "mode": "min",
                "save_last": None,
                "save_top_k": 1,
                "save_weights_only": False,
                "_every_n_train_steps": 0,
            },
        },
    )
    wandb.init().log_artifact.assert_called_with(wandb.Artifact(), aliases=["latest", "best"])


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_log_model_with_score(wandb, monkeypatch, tmp_path):
    """Test to prevent regression on #15543, ensuring the score is logged as a Python number, not a scalar tensor."""
    wandb.run = None
    model = BoringModel()

    wandb.init().log_artifact.reset_mock()
    wandb.init.reset_mock()
    wandb.Artifact.reset_mock()
    logger = WandbLogger(save_dir=tmp_path, log_model=True)
    logger.experiment.id = "1"
    logger.experiment.name = "run_name"
    checkpoint_callback = ModelCheckpoint(monitor="step")
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=1,
    )
    trainer.fit(model)

    calls = wandb.Artifact.call_args_list
    assert len(calls) == 1
    score = calls[0][1]["metadata"]["score"]
    # model checkpoint monitors scalar tensors, but wandb can't serializable them - expect Python scalars in metadata
    assert isinstance(score, int)
    assert score == 3


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_log_media(wandb, tmp_path):
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


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_logger_offline_log_model(wandb, tmp_path):
    """Test that log_model=True raises an error in offline mode."""
    with pytest.raises(MisconfigurationException, match="checkpoints cannot be uploaded in offline mode"):
        _ = WandbLogger(save_dir=str(tmp_path), offline=True, log_model=True)


@mock.patch("lightning.pytorch.loggers.wandb._WANDB_AVAILABLE", True)
@mock.patch("lightning.pytorch.loggers.wandb.Run", object)
@mock.patch("lightning.pytorch.loggers.wandb.wandb")
def test_wandb_logger_download_artifact(wandb, tmp_path):
    """Test that download_artifact works."""
    wandb.run = wandb.init()
    logger = WandbLogger()
    logger.download_artifact("test_artifact", str(tmp_path), "model", True)
    wandb.run.use_artifact.assert_called_once_with("test_artifact")

    wandb.run = None

    WandbLogger.download_artifact("test_artifact", str(tmp_path), "model", True)

    wandb.Api().artifact.assert_called_once_with("test_artifact", type="model")
