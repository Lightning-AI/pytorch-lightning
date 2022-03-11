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
import inspect
import logging
import os
import time
from collections import defaultdict
from enum import IntEnum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

import torch
from torch import nn, Tensor
from torch.autograd.profiler import record_function

from pytorch_lightning.profiler.profiler import Profiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE, _TORCH_GREATER_EQUAL_1_9
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.warnings import WarningCache

if TYPE_CHECKING:
    from torch.autograd.profiler import EventList
    from torch.utils.hooks import RemovableHandle

    from pytorch_lightning.core.lightning import LightningModule

if _KINETO_AVAILABLE:
    from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler

log = logging.getLogger(__name__)
warning_cache = WarningCache()

_PROFILER = Union[torch.autograd.profiler.profile, torch.cuda.profiler.profile, torch.autograd.profiler.emit_nvtx]


class RegisterRecordFunction:
    """While profiling autograd operations, this class will add labels for module names around the forward
    function.

    The Lightning PyTorch Profiler will activate this feature automatically. It can be deactivated as follows:

    Example::
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(record_module_names=False)
        Trainer(profiler=profiler)

    It can be used outside of Lightning as follows:

    Example::
        from pytorch_lightning import Trainer, seed_everything
        with RegisterRecordFunction(model):
            out = model(batch)
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._records: Dict[str, record_function] = {}
        self._handles: Dict[str, List["RemovableHandle"]] = {}

    def _start_recording_forward(self, _: nn.Module, input: Tensor, record_name: str) -> Tensor:
        # Add [pl][module] in name for pytorch profiler to recognize
        record = record_function("[pl][module]" + record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(self, _: nn.Module, __: Tensor, output: Tensor, record_name: str) -> Tensor:
        self._records[record_name].__exit__(None, None, None)
        return output

    def __enter__(self) -> None:
        for module_name, module in self._model.named_modules():
            if module_name:
                full_name = f"{type(module).__module__}.{type(module).__name__}"
                record_name = f"{full_name}: {module_name}"
                pre_forward_handle = module.register_forward_pre_hook(
                    partial(self._start_recording_forward, record_name=record_name)
                )
                post_forward_handle = module.register_forward_hook(
                    partial(self._stop_recording_forward, record_name=record_name)
                )

                self._handles[module_name] = [pre_forward_handle, post_forward_handle]

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}


class ScheduleWrapper:
    """This class is used to override the schedule logic from the profiler and perform recording for both
    `training_step`, `validation_step`."""

    def __init__(self, schedule: Callable) -> None:
        if not _KINETO_AVAILABLE:
            raise ModuleNotFoundError("You are trying to use `ScheduleWrapper` which require kineto install.")
        self._schedule = schedule
        self.reset()

    def setup(self, start_action_name: str) -> None:
        self._start_action_name = start_action_name

    def pre_step(self, current_action: str) -> None:
        self._current_action = current_action

    def reset(self):
        # handle properly `fast_dev_run`. PyTorch Profiler will fail otherwise.
        self._num_training_step = 0
        self._num_validation_step = 0
        self._num_test_step = 0
        self._num_predict_step = 0
        self._training_step_reached_end = False
        self._validation_step_reached_end = False
        self._test_step_reached_end = False
        self._predict_step_reached_end = False
        # used to stop profiler when `ProfilerAction.RECORD_AND_SAVE` is reached.
        self._current_action: Optional[str] = None
        self._prev_schedule_action: Optional[ProfilerAction] = None
        self._start_action_name: Optional[str] = None

    @property
    def is_training(self):
        return self._current_action.endswith("training_step")

    @property
    def is_validating(self):
        return self._current_action.endswith("validation_step")

    @property
    def is_testing(self):
        return self._current_action.endswith("test_step")

    @property
    def is_predicting(self):
        return self._current_action.endswith("predict_step")

    @property
    def num_step(self) -> int:
        if self.is_training:
            return self._num_training_step
        if self.is_validating:
            return self._num_validation_step
        if self.is_testing:
            return self._num_test_step
        if self.is_predicting:
            return self._num_predict_step
        return 0

    def _step(self) -> None:
        if self.is_training:
            self._num_training_step += 1
        elif self.is_validating:
            if self._start_action_name.endswith("on_fit_start"):
                if self._num_training_step > 0:
                    self._num_validation_step += 1
            else:
                self._num_validation_step += 1
        elif self.is_testing:
            self._num_test_step += 1
        elif self.is_predicting:
            self._num_predict_step += 1

    @property
    def has_finished(self) -> bool:
        if self.is_training:
            return self._training_step_reached_end
        if self.is_validating:
            return self._validation_step_reached_end
        if self.is_testing:
            return self._test_step_reached_end
        if self.is_predicting:
            return self._predict_step_reached_end
        return False

    def __call__(self, num_step: int) -> "ProfilerAction":
        # ignore the provided input. Keep internal state instead.
        if self._current_action is None or self.has_finished:
            return ProfilerAction.NONE

        self._step()
        action = self._schedule(max(self.num_step, 0))
        if self._prev_schedule_action == ProfilerAction.RECORD and action == ProfilerAction.WARMUP:
            # Work around the corner case when validation starts before train.
            # In this case, the action is RECORD in validation loop, and then call into the train
            # and the action is still WARMUP in train and pytorch will recognize this as error.
            action = ProfilerAction.RECORD
        if action == ProfilerAction.RECORD_AND_SAVE:
            if self.is_training:
                self._training_step_reached_end = True
            elif self.is_validating:
                self._validation_step_reached_end = True
            elif self.is_testing:
                self._test_step_reached_end = True
            elif self.is_predicting:
                self._predict_step_reached_end = True
        self._prev_schedule_action = action
        return action


class BasePyTorchProfiler(Profiler):

    STEP_FUNCTIONS = {"training_step", "validation_step", "test_step", "predict_step"}
    AVAILABLE_SORT_KEYS = {
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count",
    }

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
        """This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of.

        different operators inside your model - both on the CPU and GPU

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            row_limit: Limit the number of rows in a table, ``-1`` is a special value that
                removes the limit completely.

            sort_by_key: Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

            record_module_names: Whether to add module names while recording autograd operation.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
        """
        super().__init__(dirpath=dirpath, filename=filename)

        self._group_by_input_shapes = group_by_input_shapes and profiler_kwargs.get("record_shapes", False)
        self._emit_nvtx = emit_nvtx
        self._export_to_chrome = export_to_chrome
        self._row_limit = row_limit
        self._sort_by_key = sort_by_key or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"
        self._record_module_names = record_module_names
        self._profiler_kwargs = profiler_kwargs

        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional["EventList"] = None
        self._lightning_module: Optional["LightningModule"] = None  # set by ProfilerConnector
        self._register: Optional[RegisterRecordFunction] = None
        self._parent_profiler: Optional[_PROFILER] = None
        self._recording_map: Dict[str, record_function] = {}

        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {self._sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

    def _init_kineto_params(self, profiler_kwargs: Any) -> None:
        """Initialize the profiler arguments by filling activities and with_stack arguments."""
        activities = profiler_kwargs.get("activities", None)
        self._profiler_kwargs["activities"] = activities or self._default_activities()
        self._export_to_flame_graph = profiler_kwargs.get("export_to_flame_graph", False)
        self._metric = profiler_kwargs.get("metric", "self_cpu_time_total")
        with_stack = profiler_kwargs.get("with_stack", False) or self._export_to_flame_graph
        self._profiler_kwargs["with_stack"] = with_stack

    def _default_activities(self) -> List["ProfilerActivity"]:
        activities = []
        if not _KINETO_AVAILABLE:
            return activities
        if self._profiler_kwargs.get("use_cpu", True):
            activities.append(ProfilerActivity.CPU)
        if self._profiler_kwargs.get("use_cuda", torch.cuda.is_available()):
            activities.append(ProfilerActivity.CUDA)
        return activities

    def start(self, action_name: str) -> None:
        if self.profiler is None:
            # close profiler if it is already opened. might happen if 2 profilers
            # are created and the first one did not call `describe`
            if torch.autograd._profiler_enabled():
                torch.autograd._disable_profiler()

            self._init_profiler(action_name)

        if self._lightning_module is not None and self._register is None and self._record_module_names:
            self._register = RegisterRecordFunction(self._lightning_module)
            self._register.__enter__()

        if self.profiler is not None and action_name not in self._recording_map:

            # start profile first then record_function to allow it can be captured by the stop
            if _KINETO_AVAILABLE and not self._emit_nvtx:
                self._start_action(action_name)

            # Add [pl][profile] in name for pytorch profiler to recognize
            recording = record_function("[pl][profile]" + action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if not _KINETO_AVAILABLE or self._emit_nvtx:
            return

        self._stop_action(action_name)

    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""

        if self._export_to_chrome and not _KINETO_AVAILABLE:
            filename = f"{self.local_rank}_trace.json"
            path_to_trace = filename if self.dirpath is None else os.path.join(self.dirpath, filename)
            self.function_events.export_chrome_trace(path_to_trace)

        data = self.function_events.key_averages(group_by_input_shapes=self._group_by_input_shapes)
        table = data.table(sort_by=self._sort_by_key, row_limit=self._row_limit)

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _create_profilers(self) -> None:
        if self._emit_nvtx:
            self._parent_profiler = self._create_profiler(torch.cuda.profiler.profile)
            self.profiler = self._create_profiler(torch.autograd.profiler.emit_nvtx)
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(self._get_profiler_class())

    def _create_profiler(self, profiler: Type[_PROFILER]) -> _PROFILER:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs = {k: v for k, v in self._profiler_kwargs.items() if k in init_parameters}
        return profiler(**kwargs)

    def _cache_functions_events(self) -> None:
        if self._emit_nvtx:
            return
        self.function_events = self.profiler.events() if _KINETO_AVAILABLE else self.profiler.function_events

    def _delete_profilers(self) -> None:
        self._delete_profiler()

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None

    def teardown(self, stage: Optional[str] = None) -> None:
        self._delete_profilers()

        for k in list(self._recording_map):
            self.stop(k)
        self._recording_map = {}

        super().teardown(stage=stage)

    def _get_profiler_class(self) -> Type:
        """Get the profiler class for created profiler instance."""
        raise NotImplementedError

    def _init_profiler(self, action_name: str) -> None:
        raise NotImplementedError

    def _start_action(self, action_name: str) -> None:
        pass

    def _stop_action(self, action_name: str) -> None:
        raise NotImplementedError

    def _delete_profiler(self) -> None:
        raise NotImplementedError


class PyTorchProfilerLegacy(BasePyTorchProfiler):
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
        """This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of.

        different operators inside your model - both on the CPU and GPU

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            row_limit: Limit the number of rows in a table, ``-1`` is a special value that
                removes the limit completely.

            sort_by_key: Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

            record_module_names: Whether to add module names while recording autograd operation.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
                If arg ``schedule`` is not a ``Callable``.
                If arg ``schedule`` does not return a ``torch.profiler.ProfilerAction``.
        """
        super().__init__(
            dirpath,
            filename,
            group_by_input_shapes,
            emit_nvtx,
            export_to_chrome,
            row_limit,
            sort_by_key,
            record_module_names,
            **profiler_kwargs,
        )

        self._start_action_name: Optional[str] = None
        self._schedule: Optional[ScheduleWrapper] = None

        if _KINETO_AVAILABLE:
            self._init_kineto(profiler_kwargs)

    def _init_kineto(self, profiler_kwargs: Any) -> None:
        has_schedule = "schedule" in profiler_kwargs
        self._has_on_trace_ready = "on_trace_ready" in profiler_kwargs

        schedule = profiler_kwargs.get("schedule", None)
        if schedule is not None:
            if not isinstance(schedule, Callable):
                raise MisconfigurationException(f"Schedule should be a callable. Found: {schedule}")
            action = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise MisconfigurationException(
                    f"Schedule should return a `torch.profiler.ProfilerAction`. Found: {action}"
                )
        schedule = schedule if has_schedule else self._default_schedule()
        self._schedule = ScheduleWrapper(schedule) if schedule is not None else schedule
        self._profiler_kwargs["schedule"] = self._schedule

        self._init_kineto_params(profiler_kwargs)

    @property
    def _total_steps(self) -> int:
        trainer = self._lightning_module.trainer
        if self._schedule.is_training:
            return trainer.num_training_batches
        if self._schedule._current_action.endswith("validation_step"):
            return sum(trainer.num_val_batches) + sum(trainer.num_sanity_val_batches)
        if self._schedule._current_action.endswith("test_step"):
            return sum(trainer.num_test_batches)
        if self._schedule._current_action.endswith("predict_step"):
            return sum(trainer.num_predict_batches)

    def _should_override_schedule(self) -> bool:
        return (
            self._lightning_module is not None
            and self._schedule is not None
            and self._total_steps < 5
            and self._schedule._schedule == self._default_schedule()
        )

    @staticmethod
    @lru_cache(1)
    def _default_schedule() -> Optional[callable]:
        if _KINETO_AVAILABLE:
            # Those schedule defaults allow the profiling overhead to be negligible over training time.
            return torch.profiler.schedule(wait=1, warmup=1, active=3)

    def _get_profiler_class(self) -> Type:
        return torch.profiler.profile if _KINETO_AVAILABLE else torch.autograd.profiler.profile

    def _init_profiler(self, action_name: str) -> None:
        if self._schedule is not None:
            self._schedule.setup(action_name)

        self._create_profilers()

        profiler = self.profiler.__enter__()
        if profiler is not None:
            self.profiler = profiler

        if self._parent_profiler is not None:
            self._parent_profiler.__enter__()

    def _stop_action(self, action_name: str) -> None:
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
                        handler = tensorboard_trace_handler(
                            self.dirpath, self._prepare_filename(action_name=action_name, extension="")
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
            if _TORCH_GREATER_EQUAL_1_9:
                self.profiler.add_metadata("Framework", "pytorch-lightning")

    def _delete_profiler(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self._cache_functions_events()
            self.profiler = None

        if self._schedule is not None:
            self._schedule.reset()


class KinetoProfilerState(IntEnum):

    NONE = 0
    """PyTorch profiler is not created or is stopped."""
    WARMUP = 1
    """PyTorch profiler is created but not start."""
    START = 2
    """PyTorch profiler is running."""


class PyTorchProfilerKineto(BasePyTorchProfiler):
    """Use new lower level PyTorch profiler API torch.profiler._KinetoProfile to simplify the profiler
    implementation by removing the schedule wrapper."""

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
        wait_step: int = 1,
        warmup_step: int = 1,
        active_step: int = 3,
        **profiler_kwargs: Any,
    ) -> None:
        """This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of.

        different operators inside your model - both on the CPU and GPU

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            row_limit: Limit the number of rows in a table, ``-1`` is a special value that
                removes the limit completely.

            sort_by_key: Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

            record_module_names: Whether to add module names while recording autograd operation.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
        """
        super().__init__(
            dirpath,
            filename,
            group_by_input_shapes,
            emit_nvtx,
            export_to_chrome,
            row_limit,
            sort_by_key,
            record_module_names,
            **profiler_kwargs,
        )

        # calcuate the warmup and active step
        self._action_step_num: Dict[str, int] = defaultdict(int)
        self._wait_step: int = wait_step
        self._warmup_step: int = warmup_step
        self._active_step: int = active_step

        self._override_steps = False

        self._start_action_name: Optional[str] = None
        self._profiler_state = KinetoProfilerState.NONE
        self._action_map = {
            (KinetoProfilerState.NONE, KinetoProfilerState.WARMUP): [torch.profiler._KinetoProfile.prepare_trace],
            (KinetoProfilerState.NONE, KinetoProfilerState.START): [torch.profiler._KinetoProfile.start],
            (KinetoProfilerState.WARMUP, KinetoProfilerState.START): [torch.profiler._KinetoProfile.start_trace],
            (KinetoProfilerState.START, KinetoProfilerState.NONE): [torch.profiler._KinetoProfile.stop],
            (KinetoProfilerState.START, KinetoProfilerState.WARMUP): [
                torch.profiler._KinetoProfile.stop,
                torch.profiler._KinetoProfile.prepare_trace,
            ],
        }

        if _KINETO_AVAILABLE:
            self._init_kineto_params(profiler_kwargs)

    def _get_profiler_class(self) -> Type:
        return torch.profiler._KinetoProfile if _KINETO_AVAILABLE else torch.autograd.profiler.profile

    def _init_profiler(self, action_name: str) -> None:
        self._create_profilers()

        # the original execution order for parent_profiler is:
        # 1), child_profiler.__enter__; 2), parent_profiler.__enter__
        # I change it to parent/child to match with the exit. Please
        # raise if there are any concerns.
        if self._parent_profiler is not None:
            self._parent_profiler.__enter__()

        if not _KINETO_AVAILABLE or self._emit_nvtx:
            self.profiler.__enter__()
        else:
            if _TORCH_GREATER_EQUAL_1_9:
                self.profiler.add_metadata("Framework", "pytorch-lightning")

            self._transit_profiler(KinetoProfilerState.START)

    def _start_action(self, action_name: str) -> None:
        self._action_step_num[action_name] += 1

        if self._lightning_module is None:
            # we don't call prepare_trace/start_trace if the lightning_module is None
            # like the following scenarios
            # with pytorch_profiler.profile("a"):
            #   a = torch.ones(42)
            return

        total_steps = self.get_total_steps(action_name)
        if total_steps is None or total_steps < self.profile_steps:
            # if the total_step is None, it means the action_name is not belong to
            # any train/validation/test/prediction steps
            warning_cache.warn(
                "The PyTorch Profiler default schedule will be overridden as there is not enough "
                "steps to properly record traces."
            )
            self._override_steps = True
            # No need to call prepare_trace & start_trace here
            # since the profiler is already started when created.
            return

        step_num = self._action_step_num[action_name]
        if step_num == self._wait_step:
            # warm up
            self._transit_profiler(KinetoProfilerState.WARMUP)
        elif step_num >= self._wait_step + self._warmup_step and step_num < self.profile_steps:
            # begin profile
            self._transit_profiler(KinetoProfilerState.START)

    def _stop_action(self, action_name: str) -> None:
        if self.profiler is not None and any(action_name.endswith(func) for func in self.STEP_FUNCTIONS):
            # Save the first action name in order to save the profile data
            # when destroying profiler like what the original on_trace_ready did.
            if self._start_action_name is None:
                self._start_action_name = action_name

            if self._action_step_num[action_name] == self.profile_steps:
                self._transit_profiler(KinetoProfilerState.NONE)
                if self.dirpath is not None:
                    self._save_result(action_name)
                else:
                    rank_zero_warn("The PyTorchProfiler failed to export trace as `dirpath` is None")

    def _delete_profiler(self) -> None:
        if self.profiler is not None:
            if not _KINETO_AVAILABLE or self._emit_nvtx:
                self.profiler.__exit__(None, None, None)
            else:
                # At the moment, the profiler should already be stoped.
                if torch.autograd._profiler_enabled():
                    self._transit_profiler(KinetoProfilerState.NONE)
                    self._save_result(self._start_action_name)
            self._cache_functions_events()
            self.profiler = None

    def _save_result(self, action_name: str) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        if self._export_to_chrome:
            path = os.path.join(
                self.dirpath,
                "{}.{}.pt.trace.json".format(
                    self._prepare_filename(action_name=action_name, extension=""), int(time.time() * 1000)
                ),
            )
            self.profiler.export_chrome_trace(path)
        if self._export_to_flame_graph:
            path = os.path.join(self.dirpath, self._prepare_filename(action_name=action_name, extension=".stack"))
            self.profiler.export_stacks(path, metric=self._metric)

    @property
    def profile_steps(self):
        return self._wait_step + self._warmup_step + self._active_step

    def get_total_steps(self, action_name) -> int:
        trainer = self._lightning_module.trainer

        if action_name.endswith("training_step"):
            return trainer.num_training_batches
        if action_name.endswith("validation_step"):
            return sum(trainer.num_val_batches) + sum(trainer.num_sanity_val_batches)
        if action_name.endswith("test_step"):
            return sum(trainer.num_test_batches)
        if action_name.endswith("predict_step"):
            return sum(trainer.num_predict_batches)

    def _transit_profiler(self, new_state: KinetoProfilerState):
        action_list = self._action_map.get((self._profiler_state, new_state))
        if action_list:
            for action in action_list:
                action(self.profiler)
            self._profiler_state = new_state


if _TORCH_GREATER_EQUAL_1_9 and hasattr(torch.profiler, "_KinetoProfile"):
    PyTorchProfiler = PyTorchProfilerKineto
else:
    PyTorchProfiler = PyTorchProfilerLegacy
