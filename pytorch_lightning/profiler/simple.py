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
"""Profiler to check if there are any bottlenecks in your code."""
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from pytorch_lightning.profiler.base import BaseProfiler

log = logging.getLogger(__name__)


class SimpleProfiler(BaseProfiler):
    """
    This profiler simply records the duration of actions (in seconds) and reports
    the mean duration of each action and the total time spent over the entire training run.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        extended: bool = True,
        output_filename: Optional[str] = None,
    ) -> None:
        """
        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never started.
        """
        super().__init__(dirpath=dirpath, filename=filename, output_filename=output_filename)
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations = defaultdict(list)
        self.extended = extended
        self.start_time = time.monotonic()

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def _make_report(self) -> Tuple[list, float]:
        total_duration = time.monotonic() - self.start_time
        report = [[a, d, 100. * np.sum(d) / total_duration] for a, d in self.recorded_durations.items()]
        report.sort(key=lambda x: x[2], reverse=True)
        return report, total_duration

    def summary(self) -> str:
        sep = os.linesep
        output_string = ""
        if self._stage is not None:
            output_string += f"{self._stage.upper()} "
        output_string += f"Profiler Report{sep}"

        if self.extended:

            if len(self.recorded_durations) > 0:
                max_key = np.max([len(k) for k in self.recorded_durations.keys()])

                def log_row(action, mean, num_calls, total, per):
                    row = f"{sep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                    row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                    return row

                output_string += log_row("Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %")
                output_string_len = len(output_string)
                output_string += f"{sep}{'-' * output_string_len}"
                report, total_duration = self._make_report()
                output_string += log_row("Total", "-", "_", f"{total_duration:.5}", "100 %")
                output_string += f"{sep}{'-' * output_string_len}"
                for action, durations, duration_per in report:
                    output_string += log_row(
                        action,
                        f"{np.mean(durations):.5}",
                        f"{len(durations):}",
                        f"{np.sum(durations):.5}",
                        f"{duration_per:.5}",
                    )
        else:

            def log_row(action, mean, total):
                return f"{sep}{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

            output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
            output_string += f"{sep}{'-' * 65}"

            for action, durations in self.recorded_durations.items():
                output_string += log_row(action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}")
        output_string += sep
        return output_string
