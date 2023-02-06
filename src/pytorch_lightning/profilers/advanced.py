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
"""Profiler to check if there are any bottlenecks in your code."""
import cProfile
import io
import logging
import pstats
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from pytorch_lightning.profilers.profiler import Profiler

log = logging.getLogger(__name__)


class AdvancedProfiler(Profiler):
    """This profiler uses Python's cProfiler to record more detailed information about time spent in each function
    call recorded during a given action.

    The output is quite verbose and you should only use this if you want very detailed reports.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
    ) -> None:
        """
        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            line_count_restriction: this can be used to limit the number of functions
                reported for each action. either an integer (to select a count of lines),
                or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)

        Raises:
            ValueError:
                If you attempt to stop recording an action which was never started.
        """
        super().__init__(dirpath=dirpath, filename=filename)
        self.profiled_actions: Dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        pr.disable()

    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()
        return self._stats_to_str(recorded_stats)

    def teardown(self, stage: Optional[str]) -> None:
        super().teardown(stage=stage)
        self.profiled_actions = {}

    def __reduce__(self) -> Tuple:
        # avoids `TypeError: cannot pickle 'cProfile.Profile' object`
        return (
            self.__class__,
            (),
            dict(dirpath=self.dirpath, filename=self.filename, line_count_restriction=self.line_count_restriction),
        )
