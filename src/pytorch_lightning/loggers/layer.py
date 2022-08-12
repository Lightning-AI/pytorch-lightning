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
"""
Layer Logger
-------------
"""
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

log = logging.getLogger(__name__)

try:
    import layer
except ModuleNotFoundError:
    layer = None


class LayerLogger(Logger):
    r"""
        Log using `Layer <https://docs.app.layer.ai>`_.

    **Installation and set-up**

    Install with pip:

    .. code-block:: bash

        pip install layer

    Create a `Layer` instance:

    .. code-block:: python

        from pytorch_lightning.loggers import LayerLogger

        layer_logger = LayerLogger("[ORG/PROJECT_NAME]", "[API_KEY]")

    Pass the logger instance to the `Trainer`:

    .. code-block:: python

        trainer = Trainer(logger=layer_logger)

    Layer will log you in and init your project to track your experiment.

    **Log metrics**

    Log from :class:`~pytorch_lightning.core.module.LightningModule`:

    .. code-block:: python

        class LitMMNIST(LightningModule):
            def validation_step(self, batch, batch_idx):
                self.log("val_accuracy", acc)

    Use directly Layer:

    .. code-block:: python

        layer.log({"train/loss": loss}, step=5)

    **Log hyper-parameters**

    Save :class:`~pytorch_lightning.core.module.LightningModule` parameters:

    .. code-block:: python

        class LitMMNIST(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()


    **Log your model**

    Just return your model from your main training function. Layer will serialize/pickle your model and register to
    model catalog under your project.

    .. code-block:: python

        @layer.model("my_pl_model")
        def train():
            model = ...
            trainer = Trainer(...)
            trainer.fit(model, ...)
            return model


        train()


    **Log media**

    Log text with:

    .. code-block:: python

        # simple text
        layer_logger.log_text(key="dataset", text="mnist")

        # dictionary
        params = {"loss_type": "cross_entropy", "optimizer_type": "adamw"}
        layer_logger.log_metrics(params)

        # pandas DataFrame
        layer_logger.log_table(key="text", dataframe=df)

        # columns and data
        layer_logger.log_table(key="text", columns=["inp", "pred"], data=[["hllo", "hello"]])


    Log images with:

    .. code-block:: python

        # using tensors (`CHW`, `HWC`, `HW`), numpy arrays or PIL images
        layer_logger.log_image(key="image", image=img)

        # using file
        layer_logger.log_image(key="image", image="./pred.jpg")

        # add a step parameter to see images with slider
        layer_logger.log_image(key="image", image="./pred.jpg", step=epoch)

    Log videos with:

    .. code-block:: python

        # using tensors (`NTCHW`, `BNTCHW`)
        layer_logger.log_video(key="video", image=img)

        # using file
        layer_logger.log_video(key="video", image="./birds.mp4")


    See Also:
        - `Layer Pytorch Lightning Demo on Google Colab <https://bit.ly/pl_layer>`__
        - `Layer Documentation <https://docs.app.layer.ai>`__

    Args:
        project_name: Name of the Layer project
        api_key: Your Layer api key. You can call layer.login() if not provided.

    Raises:
        ModuleNotFoundError:
            If `layer` package is not installed.

    """

    PARAMETERS_KEY = "hyperparams"
    PREFIX_JOIN_CHAR = "-"

    _project = None

    def __init__(
        self,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None,
        prefix: str = "",
    ):
        if layer is None:
            raise ModuleNotFoundError(
                "You want to use `layer` logger which is not installed yet, install it with `pip install layer`."
            )

        super().__init__()

        self._project_name = project_name
        self._prefix = prefix
        self._api_key = api_key

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self) -> "layer":
        r"""

        Top class `layer` object. To use Layer related functions, do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.any_layer_function(...)

        """

        # Initializes the Layer Project if not initialized yet
        if self._project is None:
            # Log user in
            if self._api_key is not None:
                layer.login_with_api_key(self._api_key)

            # Init project
            self._project = layer.init(self._project_name)

        return layer

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the project and the asset name.

        Returns:
            The name of the project and the model name
        """
        if self._context:
            return f"{self._project_name}/{self._context.asset_name()}"
        else:
            return None

    @property
    def version(self) -> Optional[Union[int, str]]:
        """Gets the full version of the model (eg. `2.3`)

        Returns:
            The model version in `[major].[minor]` format if training has started otherwise returns None
        """
        from layer.contracts.asset import AssetType

        if self._context and self._context.asset_type() == AssetType.MODEL:
            return f"{self._context.train().get_version()}.{self._context.train().get_train_index()}"
        else:
            return None

    @property
    def _context(self) -> Any:
        return layer.global_context.get_active_context()

    @rank_zero_only
    def log_text(self, key: str, text: str) -> None:
        """Log text.

        :param key: Name of the parameter/metric
        :param text: Parameter/Metric value
        :return:
        """
        self.experiment.log({key: text})

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """Log hyperparameters of your experiment.

        :param params: Hyperparameters key/values
        :return:
        """
        params = _convert_params(params)
        params = _flatten_dict(params)

        parameters_key = self.PARAMETERS_KEY

        self.experiment.log({parameters_key: params})

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Log metrics to your experiment.

        :param metrics: str/float key value pairs
        :param step: If provided, a chart will be generated from the logged float metrics
        :return:
        """
        metrics = _add_prefix(metrics, self._prefix, separator=self.PREFIX_JOIN_CHAR)
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def log_image(
        self,
        key: str,
        image: Union[Any, Path, torch.Tensor],
        format: str = "CHW",
        step: Optional[int] = None,
    ) -> None:
        """Log an image to your experiment.

        :param key: Name of the image
        :param image: Image as `PIL.Image.Image`, `Path`, `npt.NDArray` or `torch.Tensor`
        :param format: Format of your array/tensor images. Can be: `CHW`, `HWC`, `HW`
        :param step: If provided, images for every step will be visible via a slider
        :return:
        """
        metrics = {key: layer.Image(image, format=format)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_video(self, key: str, video: Union[torch.Tensor, Path], fps: Union[float, int] = 4) -> None:
        """Log a video to your experiment.

        :param key: Name of your video
        :param video: Video as `torch.Tensor` in (`NTCHW`, `BNTCHW`) formats or `Path` of a video file
        :param fps: Frame per second, applicable to only torch tensor videos
        :return:
        """
        if isinstance(video, torch.Tensor):
            self.log_metrics({key: layer.Video(video=video, fps=fps)})
        else:
            self.log_metrics({key: video})

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Optional[Any] = None,
    ) -> None:
        """Log a table containing any object type (list, str, float, int, bool).

        :param key: Name of your table
        :param columns: Column names as list
        :param data: Rows as list
        :param dataframe: pandas Dataframe to be logged
        :return:
        """
        if dataframe is not None:
            self.log_metrics({key: dataframe})
        elif data is not None and columns is not None:
            try:
                import pandas as pd

                df = pd.DataFrame(columns=columns, data=data)
                self.log_metrics({key: df})
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "You need pandas installed to log table with data+columns. Install it with `pip install pandas`"
                )
        else:
            raise Exception("You should set either columns+data or dataframe parameter to log a table!")
