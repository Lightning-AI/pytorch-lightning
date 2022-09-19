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

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

from torch.autograd.profiler import record_function
from torch.profiler import ProfilerAction

from pytorch_lightning.profilers.pytorch import PyTorchProfiler, RegisterRecordFunction, ScheduleWrapper
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE

if _KINETO_AVAILABLE:
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

log = logging.getLogger(__name__)

_PROFILER = profile


class HPUProfiler(PyTorchProfiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_module_names: bool = True,
        **profiler_kwargs: Any,
    ) -> None:
        """This profiler uses PyTorch Profiler and lets you inspect the cost of.

        different operators inside your model - both on the CPU and HPU

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            row_limit: Limit the number of rows in a table, ``-1`` is a special value that
                removes the limit completely.

            sort_by_key: Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cpu_time_total``,
                ``cpu_memory_usage``, ``self_cpu_memory_usage``, ``count``.

            record_module_names: Whether to add module names while recording autograd operation.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
                If arg ``schedule`` is not a ``Callable``.
                If arg ``schedule`` does not return a ``torch.profiler.ProfilerAction``.
        """
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            group_by_input_shapes=group_by_input_shapes,
            export_to_chrome=export_to_chrome,
            row_limit=row_limit,
            sort_by_key=sort_by_key or f"{'hpu' if profiler_kwargs.get('use_hpu', False) else 'cpu'}_time_total",
            record_module_names=record_module_names,
        )

        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional["EventList"] = None
        self._lightning_module: Optional["LightningModule"] = None  # set by ProfilerConnector
        self._register: Optional[RegisterRecordFunction] = None
        self._parent_profiler: Optional[_PROFILER] = None
        self._recording_map: Dict[str, record_function] = {}
        self._start_action_name: Optional[str] = None
        self._schedule: Optional[ScheduleWrapper] = None
        self._profiler_kwargs = profiler_kwargs

        if _KINETO_AVAILABLE:
            self._init_kineto(profiler_kwargs)

        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {self._sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

        activities = profiler_kwargs.get("activities", None)
        self._profiler_kwargs["activities"] = self.profile_hpu_activities(activities)

    def profile_hpu_activities(self, activities) -> List["ProfilerActivity"]:
        if not _KINETO_AVAILABLE:
            return activities
        activities.append(ProfilerActivity.HPU)
        return activities

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if not _KINETO_AVAILABLE or self._emit_nvtx:
            return

        if self.profiler is not None and any(action_name.endswith(func) for func in self.STEP_FUNCTIONS):
            if self._schedule is not None:
                self._schedule.pre_step(action_name)

            # the default schedule requires a minimum of 5 steps to properly work: `wait=1, warmup=1, active=3`.
            # otherwise, this will raise a `segmentation fault`.
            if self._should_override_schedule():
                warning_cache.warn(
                    "The PyTorch Profiler default schedule will be overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = torch.profiler.profiler._default_schedule_fn

            def on_trace_ready(profiler):
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        file_name = re.sub(r"[^a-zA-Z0-9]+", "_", action_name)
                        handler = tensorboard_trace_handler(
                            self.dirpath, self._prepare_filename(action_name=file_name, extension="")
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        path = os.path.join(
                            self.dirpath, self._prepare_filename(action_name=action_name, extension=".stack")
                        )
                        profiler.export_stacks(path, metric=self._metric)
                else:
                    rank_zero_warn("The PyTorchProfiler failed to export trace as `dirpath` is None")

            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready

            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()
            self.profiler.add_metadata("Framework", "pytorch-lightning")

    def summary(self) -> str:
        return "Summary not supported for HPU Profiler"
