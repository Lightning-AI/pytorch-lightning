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
import torch
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, List

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

log = logging.getLogger(__name__)
_LAYER_AVAILABLE = _RequirementAvailable("layer")

try:
    import layer
except ModuleNotFoundError:
    layer = None


class LayerLogger(Logger):
    
    PARAMETERS_KEY = "hyperparams"

    def __init__(
        self,
        project_name: str,
        api_key: Optional[str]
    ):
        if layer is None:
            raise ModuleNotFoundError(
                "You want to use `layer` logger which is not installed yet, install it with `pip install layer`."
            )

        super().__init__()

        self.project_name = project_name

        if api_key is not None:
            layer.login_with_api_key(api_key)
        layer.init(project_name)

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self) -> "layer":
        return layer

    @property
    def name(self) -> Optional[str]:
        return f"{self.project_name}/{self._context.asset_name()}"

    @property
    def version(self) -> Optional[Union[int, str]]:
        from layer.contracts.asset import AssetType
        if self._context.asset_type() == AssetType.MODEL:
            return f"{self._context.train().get_version()}.{self._context.train().get_train_index()}"
        else:
            raise NotImplementedError("Dataset versions not implemented yet!")

    @property
    def _context(self) -> Optional[layer.Context]:
        context = layer.global_context.get_active_context()
        if context is None:
            raise Exception("Layer Context is only available during training!")
        else:
            return context

    @rank_zero_only
    def log_text(self, key: str, text: str) -> None:

        self.experiment.log({key: text})

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        
        parameters_key = self.PARAMETERS_KEY

        self.experiment.log({parameters_key: params})

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def log_image(self, key: str, image: Union["PIL.Image.Image", Path, "npt.NDArray[np.complex64]", torch.Tensor],
                  format: str = "CHW",
                  step: Optional[int] = None) -> None:
        metrics = {key: layer.Image(image, format=format)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_video(self, key: str, video: Union[torch.Tensor, Path], fps: Union[float, int] = 4) -> None:

        if isinstance(video, torch.Tensor):
            self.log_metrics({key: layer.Video(video=video, fps=fps)})
        else:
            self.log_metrics({key: video})

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
    ) -> None:
        if dataframe:
            self.log_metrics({key: dataframe})
        else:
            import pandas as pd
            df = pd.DataFrame(columns=columns, data=data)
            self.log_metrics({key: df})
