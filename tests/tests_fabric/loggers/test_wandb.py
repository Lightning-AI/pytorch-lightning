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
from unittest import mock

import pytest
from lightning.fabric.loggers import WandbLogger
from lightning_utilities.test.warning import no_warning_call


def test_wandb_project_name(wandb_mock):
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


def test_wandb_logger_init(wandb_mock):
    """Verify that basic functionality of wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.

    """
    # test wandb.init called when there is no W&B run
    wandb_mock.run = None
    logger = WandbLogger(
        name="test_name", save_dir="test_save_dir", version="test_id", project="test_project", resume="never"
    )
    logger.log_metrics({"acc": 1.0})
    wandb_mock.init.assert_called_once_with(
        name="test_name", dir="test_save_dir", id="test_id", project="test_project", resume="never", anonymous=None
    )
    wandb_mock.init().log.assert_called_once_with({"acc": 1.0})

    # test wandb.init called with project as name if name not provided
    wandb_mock.run = None
    wandb_mock.init.reset_mock()
    WandbLogger(project="test_project").experiment
    wandb_mock.init.assert_called_once_with(
        name=None, dir=".", id=None, project="test_project", resume="allow", anonymous=None
    )

    # test wandb.init set save_dir correctly after created
    wandb_mock.run = None
    wandb_mock.init.reset_mock()
    logger = WandbLogger()
    assert logger.save_dir is not None
    wandb_mock.run = None
    wandb_mock.init.reset_mock()
    logger = WandbLogger(save_dir=".", dir=None)
    assert logger.save_dir is not None

    # test wandb.init and setting logger experiment externally
    wandb_mock.run = None
    run = wandb_mock.init()
    logger = WandbLogger(experiment=run)
    assert logger.experiment

    # test wandb.init not called if there is a W&B run
    wandb_mock.init().log.reset_mock()
    wandb_mock.init.reset_mock()
    wandb_mock.run = wandb_mock.init()

    logger = WandbLogger()
    with pytest.warns(UserWarning, match="There is a wandb run already in progress"):
        _ = logger.experiment

    # check that no new run is created
    with no_warning_call(UserWarning, match="There is a wandb run already in progress"):
        _ = logger.experiment

    # verify default resume value
    assert logger._wandb_init["resume"] == "allow"

    logger.log_metrics({"acc": 1.0}, step=3)
    wandb_mock.init.assert_called_once()
    wandb_mock.init().log.assert_called_once_with({"acc": 1.0, "trainer/global_step": 3})

    # continue training on same W&B run and offset step
    logger.finalize("success")
    logger.log_metrics({"acc": 1.0}, step=6)
    wandb_mock.init().log.assert_called_with({"acc": 1.0, "trainer/global_step": 6})

    # log hyper parameters
    hparams = {"test": None, "nested": {"a": 1}, "b": [2, 3, 4]}
    logger.log_hyperparams(hparams)
    wandb_mock.init().config.update.assert_called_once_with(hparams, allow_val_change=True)

    assert logger.version == wandb_mock.init().id


def test_wandb_logger_init_before_spawn(wandb_mock):
    logger = WandbLogger()
    assert logger._experiment is None
    logger.__getstate__()
    assert logger._experiment is not None


def test_wandb_log_media(wandb_mock, tmp_path):
    """Test that the logger creates the folders and files in the right place."""
    wandb_mock.run = None

    # test log_text with columns and data
    columns = ["input", "label", "prediction"]
    data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]
    logger = WandbLogger()
    logger.log_text(key="samples", columns=columns, data=data)
    wandb_mock.Table.assert_called_once_with(
        columns=["input", "label", "prediction"],
        data=[["cheese", "english", "english"], ["fromage", "french", "spanish"]],
        dataframe=None,
    )
    wandb_mock.init().log.assert_called_once_with({"samples": wandb_mock.Table()})

    # test log_text with dataframe
    wandb_mock.Table.reset_mock()
    wandb_mock.init().log.reset_mock()
    df = 'pandas.DataFrame({"col1": [1, 2], "col2": [3, 4]})'  # TODO: incompatible numpy/pandas versions in test env
    logger.log_text(key="samples", dataframe=df)
    wandb_mock.Table.assert_called_once_with(
        columns=None,
        data=None,
        dataframe=df,
    )
    wandb_mock.init().log.assert_called_once_with({"samples": wandb_mock.Table()})

    # test log_image
    wandb_mock.init().log.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"])
    wandb_mock.Image.assert_called_with("2.jpg")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Image(), wandb_mock.Image()]})

    # test log_image with step
    wandb_mock.init().log.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"], step=5)
    wandb_mock.Image.assert_called_with("2.jpg")
    wandb_mock.init().log.assert_called_once_with(
        {"samples": [wandb_mock.Image(), wandb_mock.Image()], "trainer/global_step": 5}
    )

    # test log_image with captions
    wandb_mock.init().log.reset_mock()
    wandb_mock.Image.reset_mock()
    logger.log_image(key="samples", images=["1.jpg", "2.jpg"], caption=["caption 1", "caption 2"])
    wandb_mock.Image.assert_called_with("2.jpg", caption="caption 2")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Image(), wandb_mock.Image()]})

    # test log_image without a list
    with pytest.raises(TypeError, match="""Expected a list as "images", found <class 'str'>"""):
        logger.log_image(key="samples", images="1.jpg")

    # test log_image with wrong number of captions
    with pytest.raises(ValueError, match="Expected 2 items but only found 1 for caption"):
        logger.log_image(key="samples", images=["1.jpg", "2.jpg"], caption=["caption 1"])

    # test log_audio
    wandb_mock.init().log.reset_mock()
    logger.log_audio(key="samples", audios=["1.mp3", "2.mp3"])
    wandb_mock.Audio.assert_called_with("2.mp3")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Audio(), wandb_mock.Audio()]})

    # test log_audio with step
    wandb_mock.init().log.reset_mock()
    logger.log_audio(key="samples", audios=["1.mp3", "2.mp3"], step=5)
    wandb_mock.Audio.assert_called_with("2.mp3")
    wandb_mock.init().log.assert_called_once_with(
        {"samples": [wandb_mock.Audio(), wandb_mock.Audio()], "trainer/global_step": 5}
    )

    # test log_audio with captions
    wandb_mock.init().log.reset_mock()
    wandb_mock.Audio.reset_mock()
    logger.log_audio(key="samples", audios=["1.mp3", "2.mp3"], caption=["caption 1", "caption 2"])
    wandb_mock.Audio.assert_called_with("2.mp3", caption="caption 2")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Audio(), wandb_mock.Audio()]})

    # test log_audio without a list
    with pytest.raises(TypeError, match="""Expected a list as "audios", found <class 'str'>"""):
        logger.log_audio(key="samples", audios="1.mp3")

    # test log_audio with wrong number of captions
    with pytest.raises(ValueError, match="Expected 2 items but only found 1 for caption"):
        logger.log_audio(key="samples", audios=["1.mp3", "2.mp3"], caption=["caption 1"])

    # test log_video
    wandb_mock.init().log.reset_mock()
    logger.log_video(key="samples", videos=["1.mp4", "2.mp4"])
    wandb_mock.Video.assert_called_with("2.mp4")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Video(), wandb_mock.Video()]})

    # test log_video with step
    wandb_mock.init().log.reset_mock()
    logger.log_video(key="samples", videos=["1.mp4", "2.mp4"], step=5)
    wandb_mock.Video.assert_called_with("2.mp4")
    wandb_mock.init().log.assert_called_once_with(
        {"samples": [wandb_mock.Video(), wandb_mock.Video()], "trainer/global_step": 5}
    )

    # test log_video with captions
    wandb_mock.init().log.reset_mock()
    wandb_mock.Video.reset_mock()
    logger.log_video(key="samples", videos=["1.mp4", "2.mp4"], caption=["caption 1", "caption 2"])
    wandb_mock.Video.assert_called_with("2.mp4", caption="caption 2")
    wandb_mock.init().log.assert_called_once_with({"samples": [wandb_mock.Video(), wandb_mock.Video()]})

    # test log_video without a list
    with pytest.raises(TypeError, match="""Expected a list as "videos", found <class 'str'>"""):
        logger.log_video(key="samples", videos="1.mp4")

    # test log_video with wrong number of captions
    with pytest.raises(ValueError, match="Expected 2 items but only found 1 for caption"):
        logger.log_video(key="samples", videos=["1.mp4", "2.mp4"], caption=["caption 1"])

    # test log_table
    wandb_mock.Table.reset_mock()
    wandb_mock.init().log.reset_mock()
    logger.log_table(key="samples", columns=columns, data=data, dataframe=df, step=5)
    wandb_mock.Table.assert_called_once_with(
        columns=columns,
        data=data,
        dataframe=df,
    )
    wandb_mock.init().log.assert_called_once_with({"samples": wandb_mock.Table(), "trainer/global_step": 5})


def test_wandb_logger_download_artifact(wandb_mock, tmp_path):
    """Test that download_artifact works."""
    wandb_mock.run = wandb_mock.init()
    logger = WandbLogger()
    logger.download_artifact("test_artifact", str(tmp_path), "model", True)
    wandb_mock.run.use_artifact.assert_called_once_with("test_artifact")

    wandb_mock.run = None

    WandbLogger.download_artifact("test_artifact", str(tmp_path), "model", True)

    wandb_mock.Api().artifact.assert_called_once_with("test_artifact", type="model")
