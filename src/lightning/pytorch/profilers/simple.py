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
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from lightning.pytorch.profilers.profiler import Profiler

log = logging.getLogger(__name__)

_TABLE_ROW_EXTENDED = Tuple[str, float, int, float, float]
_TABLE_DATA_EXTENDED = List[_TABLE_ROW_EXTENDED]
_TABLE_ROW = Tuple[str, float, float]
_TABLE_DATA = List[_TABLE_ROW]


class SimpleProfiler(Profiler):
    """This profiler simply records the duration of actions (in seconds) and reports the mean duration of each
    action and the total time spent over the entire training run."""

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        extended: bool = True,
    ) -> None:
        """
        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~lightning.pytorch.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            extended: If ``True``, adds extra columns representing number of calls and percentage of total time spent on
                respective action.

        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never started.
        """
        super().__init__(dirpath=dirpath, filename=filename)
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations: Dict = defaultdict(list)
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

    def _make_report_extended(self) -> Tuple[_TABLE_DATA_EXTENDED, float, float]:
        total_duration = time.monotonic() - self.start_time
        report = [
            (a, np.mean(d), len(d), np.sum(d), 100.0 * np.sum(d) / total_duration)
            for a, d in self.recorded_durations.items()
        ]
        report.sort(key=lambda x: x[4], reverse=True)
        total_calls = sum(x[2] for x in report)
        return report, total_calls, total_duration

    def _make_report(self) -> _TABLE_DATA:
        report = [(action, np.mean(d), np.sum(d)) for action, d in self.recorded_durations.items()]
        report.sort(key=lambda x: x[1], reverse=True)
        return report

    def summary(self) -> str:
        sep = os.linesep
        output_string = ""
        if self._stage is not None:
            output_string += f"{self._stage.upper()} "
        output_string += f"Profiler Report{sep}"

        if self.extended:

            if len(self.recorded_durations) > 0:
                max_key = max(len(k) for k in self.recorded_durations.keys())

                def log_row_extended(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
                    row = f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|"
                    row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                    return row

                header_string = log_row_extended(
                    "Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %"
                )
                output_string_len = len(header_string.expandtabs())
                sep_lines = f"{sep}{'-' * output_string_len}"
                output_string += sep_lines + header_string + sep_lines
                report_extended: _TABLE_DATA_EXTENDED
                report_extended, total_calls, total_duration = self._make_report_extended()
                output_string += log_row_extended("Total", "-", f"{total_calls:}", f"{total_duration:.5}", "100 %")
                output_string += sep_lines
                for action, mean_duration, num_calls, total_duration, duration_per in report_extended:
                    output_string += log_row_extended(
                        action,
                        f"{mean_duration:.5}",
                        f"{num_calls}",
                        f"{total_duration:.5}",
                        f"{duration_per:.5}",
                    )
                output_string += sep_lines
        else:
            max_key = max(len(k) for k in self.recorded_durations)

            def log_row(action: str, mean: str, total: str) -> str:
                return f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|  {total:<15}\t|"

            header_string = log_row("Action", "Mean duration (s)", "Total time (s)")
            output_string_len = len(header_string.expandtabs())
            sep_lines = f"{sep}{'-' * output_string_len}"
            output_string += sep_lines + header_string + sep_lines
            report = self._make_report()

            for action, mean_duration, total_duration in report:
                output_string += log_row(action, f"{mean_duration:.5}", f"{total_duration:.5}")
            output_string += sep_lines
        output_string += sep
        return output_string
