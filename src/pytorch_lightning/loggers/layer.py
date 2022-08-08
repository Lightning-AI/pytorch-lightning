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
import os
import re
from argparse import Namespace
from time import time
from typing import Any, Dict, Mapping, Optional, Union

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.imports import _module_available
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)
LOCAL_FILE_URI_PREFIX = "file:"
_MLFLOW_AVAILABLE = _module_available("layer")

try:
    import layer
except ModuleNotFoundError:
    _LAYER_AVAILABLE = False


class LayerLogger(Logger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        project_name: str,
        api_key: str
    ):
        if layer is None:
            raise ModuleNotFoundError(
                "You want to use `layer` logger which is not installed yet, install it with `pip install layer`."
            )

        super().__init__()

        self.project_name = project_name

        layer.login_with_api_key(api_key)
        layer.init(project_name)

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self) -> layer:
        return layer

    @property
    def run_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the run id.

        Returns:
            The run id.
        """
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the experiment id.

        Returns:
            The experiment id.
        """
        _ = self.experiment
        return self._experiment_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)

        self.experiment.log(params)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def finalize(self, status: str = "FINISHED") -> None:
        super().finalize(status)

    @property
    def save_dir(self) -> Optional[str]:
        """The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwise returns `None`.
        """
        return None

    @property
    def name(self) -> Optional[str]:
        """Get the experiment id.

        Returns:
            The experiment id.
        """
        return self.project_name

    @property
    def version(self) -> Optional[str]:
        """Get the run id.

        Returns:
            The run id.
        """
        return self.project_name
