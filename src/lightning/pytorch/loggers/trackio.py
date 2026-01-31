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

from argparse import Namespace
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.utilities.logger import (
    _convert_json_serializable,
    _convert_params,
    _sanitize_callable_params,
)
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from trackio import Run

_TRACKIO_AVAILABLE = RequirementCache("trackio")


class TrackioLogger(Logger):
    r"""Log metrics and hyperparameters to `Trackio <https://huggingface.co/docs/trackio/en/index>`_.

    Args:
        project: The name of the project to which the experiment belongs.
        name: The name of the run. If not provided, it defaults to the project name
        resume: Resume behavior, one of 'never', 'allow', or 'must'. Defaults to 'allow'.
        **kwargs: Additional keyword arguments passed to `trackio.init()`.

    Raises:
        ModuleNotFoundError: If trackio is not installed.

    Example:
        .. testcode::
            :skipif: not _TRACKIO_AVAILABLE

            from lightning.pytorch.loggers import TrackioLogger
            from lightning.pytorch import Trainer

            trackio_logger = TrackioLogger(
                project="my_project",
                name="my_experiment",
            )
            trainer = Trainer(max_epochs=10, logger=trackio_logger)

    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        resume: Literal["never", "allow", "must"] = "allow",
        **kwargs: Any,
    ):
        if not _TRACKIO_AVAILABLE:
            raise ModuleNotFoundError(str(_TRACKIO_AVAILABLE))
        super().__init__()
        self._project = project
        self._name = name

        kwargs["resume"] = resume
        self._kwargs = kwargs
        self._experiment: Run = None

    @property
    def name(self) -> str:
        return self._project

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._name if self._experiment is None else self.experiment.name

    @property
    @rank_zero_experiment
    def experiment(self) -> "Run":
        if self._experiment is None:
            import trackio

            run = trackio.init(
                project=self._project,
                name=self._name,
                **self._kwargs,
            )
            self._experiment = run
        return self._experiment

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = _convert_json_serializable(params)
        self.experiment.config.update(params)

    @override
    @rank_zero_only
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        self.experiment.log(metrics, step=step)

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._finish_experiment()

    def _finish_experiment(self) -> None:
        if self.experiment is not None and not self.experiment._stop_flag.is_set():
            self.experiment.finish()

    def __del__(self) -> None:
        self._finish_experiment()
