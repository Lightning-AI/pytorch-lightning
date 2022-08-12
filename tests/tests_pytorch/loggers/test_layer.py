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

from unittest import mock

from pytorch_lightning.loggers import LayerLogger

project_name = "test_project"
api_key = "test_api_key"


@mock.patch("pytorch_lightning.loggers.layer.layer")
def test_layer_logger_init(layer):
    """Check if Layer is initialized correctly."""

    # Test if Layer login and init is called
    logger = LayerLogger(project_name=project_name, api_key=api_key)
    logger.log_metrics({"acc": 1.0})

    layer.login_with_api_key.assert_called_once_with(api_key)
    layer.init.assert_called_once_with("test_project")
    layer.log.assert_called_once_with({"acc": 1.0}, step=None)


@mock.patch("pytorch_lightning.loggers.layer.layer")
def test_layer_log_media(layer):
    """Check if media is logged correctly."""

    logger = LayerLogger(project_name, api_key)

    # Check logging simple text
    layer.log.reset_mock()
    logger.log_text(key="text", text="value")
    layer.log.assert_called_once_with({"text": "value"})

    # Check logging a dataframe
    columns = ["col1"]
    rows = [["val1"]]
    layer.log.reset_mock()
    import pandas as pd

    df = pd.DataFrame(columns=columns, data=rows)
    logger.log_table(key="dataframe", dataframe=df)
    args, kwargs = layer.log.call_args
    assert df.equals(args[0]["dataframe"])

    # Check logging a table
    layer.log.reset_mock()
    logger.log_table(key="table", columns=columns, data=rows)
    args, kwargs = layer.log.call_args
    assert df.equals(args[0]["table"])

    # Check logging an image
    layer.log.reset_mock()
    img = "1.jpg"
    logger.log_image(key="image", image=img)
    layer.log.assert_called_once_with({"image": layer.Image(img)}, step=None)

    # Check logging stepped image
    layer.log.reset_mock()
    logger.log_image(key="image", image=img, step=10)
    layer.log.assert_called_once_with({"image": layer.Image(img)}, step=10)

    # Check logging a video
    layer.log.reset_mock()
    video = "test.mp4"
    logger.log_video(key="video", video=video)
    layer.log.assert_called_once_with({"video": video}, step=None)

    # Check logging torch tensors as video
    import torch

    layer.log.reset_mock()
    video_tensor = torch.rand(10, 3, 100, 200)
    logger.log_video(key="video", video=video_tensor, fps=4)
    layer.log.assert_called_once_with({"video": layer.Video(video_tensor, fps=4)}, step=None)
