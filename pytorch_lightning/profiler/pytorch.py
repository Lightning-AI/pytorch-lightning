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
from functools import partial
from typing import Callable, List, Optional, Any

import torch

from pytorch_lightning.profiler.profilers import BaseProfiler
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_8, rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_8:
    from torch.autograd.profiler import record_function
    from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler

log = logging.getLogger(__name__)


class LegacyPyTorchProfiler(BaseProfiler):

    RECORD_FUNCTIONS = ("training_step_and_backward", "training_step", "backward", "validation_step", "test_step")
    AVAILABLE_SORT_KEYS = (
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count",
    )

    def __init__(
        self,
        output_filename: Optional[str] = None,
        enabled: bool = True,
        use_cpu: bool = True,
        use_cuda: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = False,
        group_by_input_shapes: bool = False,
        with_stack: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = False,
        path_to_export_trace: Optional[str] = None,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: Optional[List[str]] = None,
        local_rank: Optional[int] = None,
        profiled_functions: Optional[List[str]] = None,
    ) -> None:
        """
        This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
        different operators inside your model - both on the CPU and GPU

        Args:
            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished. When using ``ddp``,
                each rank will stream the profiled operation to their own file
                with the extension ``_{rank}.txt``

            enabled: Setting this to False makes this context manager a no-op.

            use_cpu: record events on the CPU

            use_cuda: Enables timing of CUDA events as well using the cudaEvent API.
                Adds approximately 4us of overhead to each tensor operation.

            record_shapes: If shapes recording is set, information about input dimensions will be collected.

            profile_memory: Whether to report memory usage (introduced in PyTorch 1.6.0)

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            with_stack: record source information (file and line number) for the ops (Introduced in PyTorch 1.7.0)

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            path_to_export_trace: Directory path to export ``.json`` traces when using ``export_to_chrome=True``.
                By default, it will be save where the file being is being run.

            row_limit: Limit the number of rows in a table, ``0`` is a special value that
                removes the limit completely.

            sort_by_key: Keys to sort out profiled table.

            record_functions: list of profiled functions which will create a context manager on.
                Any other will be pass through.

            local_rank: When running in distributed setting, local_rank is used for each process
                to write to their own file if `output_fname` is provided.

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``, or
                if log file is not a ``.txt`` file.
            ValueError:
                If you attempt to stop recording an action which was never started.
        """
        if profiled_functions is not None:
            # TODO: DeprecationWarning
            if record_functions is None:
                record_functions = profiled_functions
            else:
                # TODO; raise MisconfigurationException
                pass

        self.enabled = enabled
        self.record_functions = record_functions or self.RECORD_FUNCTIONS
        self.use_cuda = use_cuda
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.sort_by_key = sort_by_key or ("cuda_time_total" if self.use_cuda else "cpu_time_total")
        self.with_stack = with_stack
        self.group_by_input_shapes = group_by_input_shapes and record_shapes
        self.use_cpu = use_cpu
        self.row_limit = row_limit
        self.emit_nvtx = emit_nvtx
        self.export_to_chrome = export_to_chrome
        self.path_to_export_trace = path_to_export_trace

        if export_to_chrome and path_to_export_trace is None:
            rank_zero_warn(
                "The exported trace would be save locally as `path_to_export_trace` is empty."
                " Note: Each functions will generate its own traced file."
            )

        if self.sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

        self.profiled_actions = {}
        self.context_names = {}
        self.running_stack = []
        self.profiler = None

        self.output_fname = output_filename
        self.output_file = None
        if local_rank is not None:
            self.on_train_start(local_rank=local_rank)
            self.on_train_start = super().on_train_start

    def on_train_start(self, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
        """
        This function is used by the Trainer to inject local_rank with `DDP`
        and `TensorBoardLogger` log_dir in the profiler.
        """
        self.local_rank = local_rank

        # if the user didn't `path_to_export_trace`,
        # set it as TensorBoardLogger log_dir if exists

        if self.path_to_export_trace is None:
            self.path_to_export_trace = log_dir

        # when logging to `log.info`, only perform profiling on rank 0
        if local_rank != 0 and self.output_fname is None:
            self.wrap_functions_into_rank_zero_only()

        if self.output_fname:
            if local_rank is not None:
                if '.txt' not in self.output_fname:
                    raise MisconfigurationException("Log file should be .txt file.")

                self.output_fname = self.output_fname.replace(".txt", f"_{self.local_rank}.txt")

            fs = get_filesystem(self.output_fname)
            self.output_file = fs.open(self.output_fname, "w")

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
        super().__init__(output_streams=streaming_out)

    def wrap_functions_into_rank_zero_only(self) -> None:
        # TODO: what's the point of wrapping like this?
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)
        self.describe = rank_zero_only(self.describe)

    def start(self, action_name: str) -> None:
        if action_name not in self.record_functions:
            return

        if len(self.running_stack) > 0:
            self._stop(self.running_stack[-1])
        self.running_stack.append(action_name)

        self.context_names[action_name] = "/".join(self.running_stack)

        self._start()

    def _start(self) -> None:
        if self.emit_nvtx:
            self._create_profiler(torch.cuda.profiler.profile, enter=False)
            self._create_profiler(torch.autograd.profiler.emit_nvtx)
        else:
            self._create_profiler(torch.autograd.profiler.profile)

    def _create_profiler(self, profiler, enter=True):
        init_args = inspect.signature(profiler.__init__).parameters
        profiler_args = {k: v for k, v in vars(self).items() if k in init_args}
        pr = profiler(**profiler_args)
        if enter:
            pr = pr.__enter__()
        self.profiler = pr

    @property
    def function_events(self):
        return self.profiler.function_events

    def _stop(self, triggered_by_stop_function: bool = False) -> None:
        if self.profiler is None:
            return

        self.profiler.__exit__(None, None, None)

        function_events = self.function_events
        if not triggered_by_stop_function:
            self.profiler = None
        if function_events is not None:
            for name in self.running_stack:
                if name not in self.profiled_actions:
                    self.profiled_actions[name] = function_events
                else:
                    self.profiled_actions[name] += function_events

    def stop(self, action_name: str) -> None:
        if action_name not in self.record_functions:
            return

        if len(self.running_stack) == 0 or self.running_stack[-1] != action_name:
            raise ValueError(  # pragma: no-cover
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        self._stop(triggered_by_stop_function=True)

        self.profiler = None
        self.running_stack.pop()
        # restore running profiler
        if len(self.running_stack) > 0:
            self._start()

    def _summary(self, local_rank: int) -> dict:
        recorded_stats = {}
        for action_name, function_events in self.profiled_actions.items():

            # next line is a workaround for a pytorch issue (fixed on master, still present
            # on 1.7). Without it the code fails with `AssertionError: There is already a CPU
            # parent event for detach`
            function_events.populate_cpu_children = lambda: None

            if self.export_to_chrome:
                filename = f"{action_name}_{local_rank}_trace.json"
                path_to_trace = (
                    filename
                    if self.path_to_export_trace is None else
                    os.path.join(self.path_to_export_trace, filename)
                )
                function_events.export_chrome_trace(path_to_trace)

            if self.emit_nvtx:
                return recorded_stats
            data = function_events.key_averages(group_by_input_shapes=self.group_by_input_shapes)
            table = data.table(sort_by=self.sort_by_key, row_limit=self.row_limit)
            recorded_stats[action_name] = table
        return recorded_stats

    def summary(self) -> str:
        if not self.enabled:
            return ""

        local_rank = 0 if self.local_rank is None else self.local_rank
        recorded_stats = self._summary(local_rank)
        if not recorded_stats:
            return ""

        output = ["Profiler Report"]
        for action, stats in recorded_stats.items():
            output.append(f"Profile stats for: {action} rank: {local_rank}")
            output.append(stats)
        return os.linesep.join(output)

    def describe(self) -> None:
        """Logs a profile report after the conclusion of the training run."""
        super().describe()
        if self.output_file:
            self.output_file.flush()

    def __del__(self) -> None:
        """Close profiler's stream."""
        if self.output_file:
            self.output_file.close()


class PyTorchProfiler(LegacyPyTorchProfiler):
    """
    This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
    different operators inside your model - both on the CPU and GPU.
    Available with PyTorch>=1.8, the profiler relies on the
    `PyTorch Kineto project <https://github.com/pytorch/kineto>`__.

    The PyTorch profiler API changed in 1.8, therefore this documentation will display both versions.

    Args:
        output_filename: optionally save profile results to file instead of printing
            to std out when training is finished. When using ``ddp``,
            each rank will stream the profiled operation to their own file
            with the extension ``_{rank}.txt``

        enabled: Setting this to False makes this context manager a no-op.

        use_cpu: Enables timing of CPU events.

        use_cuda: Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.

        record_shapes: If shapes recording is set, information about input dimensions will be collected.

        profile_memory: Whether to report memory usage (introduced in PyTorch 1.6)

        group_by_input_shapes: Include operator input shapes and group calls by shape.

        with_stack: record source information (file and line number) for the ops (introduced in PyTorch 1.7)

        row_limit: Limit the number of rows in a table, ``0`` is a special value that
            removes the limit completely.

        export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
            It will generate a ``.json`` file which can be read by Chrome.

        path_to_export_trace: Directory path to export ``.json`` traces when using ``export_to_chrome=True``.
            Before PyTorch 1.8, it will be save where the file being is being run.
            After PyTorch 1.8, it will save in the ``lightning_logs/version_{}`` folder.

        sort_by_key: Keys to sort out profiled table

        record_functions: list of profiled functions which will create a context manager on.
            Any other will be pass through.

        local_rank: When running in distributed setting, local_rank is used for each process
            to write to their own file if ``output_fname`` is provided.

        emit_nvtx: warning - Supported only for torch<1.8)
            Run::

                nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

            To visualize, you can either use::

                nvvp trace_name.prof
                torch.autograd.profiler.load_nvprof(path)

        export_to_flame_graph: warning - Supported only for torch>=1.8
            Whether to export the sequence of profiled operators for Flame Graph.

        on_trace_ready: warning - Supported only for torch>=1.8
            Function which takes the profiler and executed on ``RECORD_AND_SAVE`` action

        schedule: warning - Supported only for torch>=1.8
            Optional Callable which describes recording procedure
    """


if _TORCH_GREATER_EQUAL_1_8:
    class RegisterRecordFunction:
        """
        While profiling autograd operations, this class will add labels
        with module names around the forward function.
        The PyTorch Profiler will use this feature automatically.

        It can be deactivated as follows:

        Example::

            from pytorch_lightning.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(record_module_names=False)

            Trainer(profiler=profiler)

        It can be used outside of Lightning as follows:

        Example::

            with RegisterRecordFunction(model):
                out = model(batch)

        """
        NN_MODULES: List[str] = dir(torch.nn)

        def __init__(self, model) -> None:
            self._model = model
            self._records = {}
            self._handles = {}

        def _start_recording(self, input: Any, key: Optional[str] = None) -> Any:
            if key is None:
                raise ValueError
            self._records[key] = record_function(key).__enter__()
            return input

        def _stop_recording(self, result: Any, key: Optional[str] = None) -> Any:
            if key is None:
                raise ValueError
            self._records[key].__exit__(None, None, None)
            return result

        def __enter__(self):
            for module_name, module in self._model.named_modules():
                key = module_name if module in self.NN_MODULES else f"{type(module)}: {module_name}"
                pre_handle = module.register_forward_pre_hook(partial(self._start_recording, key=key))
                post_handle = module.register_forward_hook(partial(self._stop_recording, key=key))
                self._handles[module_name] = [pre_handle, post_handle]

        def __exit__(self, *_):
            self._records = {}
            self._handles = {}

    class ScheduleWrapper:
        """
        This class is used to override the schedule logic from the profiler and perform
        recording for both `training_step`, `validation_step`.
        """

        def __init__(self, schedule: Callable) -> None:
            self._schedule = schedule
            self._num_training_step_and_backward = 0
            self._num_validation_step = 0
            self._training_step_and_backward_reached_end = False
            self._validation_step_reached_end = False
            # used to stop profiler when `ProfilerAction.RECORD_AND_SAVE` is reached.
            self._none_action = None
            self._current_action = None

        @property
        def num_step(self) -> int:
            if self._current_action == "training_step_and_backward":
                return self._num_training_step_and_backward
            elif self._current_action == "validation_step":
                return self._num_validation_step
            else:
                return 0

        def _step(self) -> None:
            if self._current_action == "training_step_and_backward":
                self._num_training_step_and_backward += 1
            elif self._current_action == "validation_step" and self._num_training_step_and_backward > 0:
                # skip sanity check
                self._num_validation_step += 1

        @property
        def has_finished(self) -> bool:
            if self._current_action == "training_step_and_backward":
                return self._training_step_and_backward_reached_end
            elif self._current_action == "validation_step":
                return self._validation_step_reached_end
            return False

        def __call__(self, num_step: int) -> 'ProfilerAction':
            # ignore the provided input. Keep internal state instead.
            if self.has_finished:
                return ProfilerAction.NONE

            self._step()
            action = self._schedule(self.num_step)
            if action == ProfilerAction.RECORD_AND_SAVE:
                if self._current_action == "training_step_and_backward":
                    self._training_step_and_backward_reached_end = True
                elif self._current_action == "validation_step":
                    self._validation_step_reached_end = True
            return action

    class PyTorchProfiler(LegacyPyTorchProfiler):  # noqa F811

        START_ACTION = "on_fit_start"
        STEP_FUNCTIONS = ("training_step_and_backward", "validation_step")

        def __init__(
            self,
            output_filename: Optional[str] = None,
            enabled: bool = True,
            use_cpu: bool = True,
            use_cuda: bool = True,
            record_shapes: bool = True,
            profile_memory: bool = True,
            group_by_input_shapes: bool = False,
            with_stack: bool = False,
            export_to_chrome: bool = True,
            path_to_export_trace: Optional[str] = None,
            row_limit: int = 20,
            sort_by_key: Optional[str] = None,
            record_functions: Optional[List] = None,
            local_rank: Optional[int] = None,
            schedule: Optional[Callable] = torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready: Optional[Callable] = None,
            export_to_flame_graph: bool = True,
            with_flops: bool = True,
            metric: str = 'self_cpu_time_total',
            record_module_names: bool = True,
        ):
            """
            This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
            different operators inside your model - both on the CPU and GPU.
            Available with PyTorch>=1.8, the profiler relies on the
            `PyTorch Kineto project <https://github.com/pytorch/kineto>`__.

            The PyTorch profiler API changed in 1.8, therefore this documentation will display both versions.

            Args:
                output_filename: optionally save profile results to file instead of printing
                    to std out when training is finished. When using ``ddp``,
                    each rank will stream the profiled operation to their own file
                    with the extension ``_{rank}.txt``

                enabled: Setting this to False makes this context manager a no-op.

                use_cpu: Enables timing of CPU events.

                use_cuda: Enables timing of CUDA events as well using the cudaEvent API.
                    Adds approximately 4us of overhead to each tensor operation.

                record_shapes: If shapes recording is set, information about input dimensions will be collected.

                profile_memory: Whether to report memory usage

                group_by_input_shapes: Include operator input shapes and group calls by shape.

                with_stack: record source information (file and line number) for the ops (introduced in PyTorch 1.7)

                export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                    It will generate a ``.json`` file which can be read by Chrome.

                path_to_export_trace: Directory path to export ``.json`` traces when using ``export_to_chrome=True``.
                    It will save in the ``lightning_logs/version_{}`` folder.

                row_limit: Limit the number of rows in a table, ``0`` is a special value that
                    removes the limit completely.

                sort_by_key: Keys to sort out profiled table

                record_functions: list of profiled functions which will create a context manager on.
                    Any other will be pass through.

                local_rank: When running in distributed setting, local_rank is used for each process
                    to write to their own file if ``output_fname`` is provided.

                schedule: Optional Callable which describes recording procedure

                on_trace_ready: Function which takes the profiler and executed on ``RECORD_AND_SAVE`` action

                export_to_flame_graph: Whether to export the sequence of profiled operators for Flame Graph.

                with_flops: TODO

                metric: TODO

                record_module_names: TODO
            """
            if schedule is not None:
                if not isinstance(schedule, Callable):
                    raise MisconfigurationException(f"Found schedule: {schedule}. Schedule should be a callable.")
                action = schedule(0)
                if not isinstance(action, ProfilerAction):
                    raise MisconfigurationException(
                        f"Found schedule: {schedule}. "
                        "Schedule should be a callable returning `torch.profiler.ProfilerAction`. "
                    )

            if isinstance(sort_by_key, str) and sort_by_key not in self.AVAILABLE_SORT_KEYS:
                raise MisconfigurationException(
                    f"Found sort_by_key: {sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
                )

            self.output_filename = output_filename
            self.enabled = enabled
            self.use_cpu = use_cpu
            self.use_cuda = use_cuda and torch.cuda.is_available()
            self.record_functions = record_functions or self.RECORD_FUNCTIONS
            self.record_functions_managers = {}
            self.activities = [ProfilerActivity.CPU] * use_cpu + [ProfilerActivity.CUDA] * self.use_cuda
            self.schedule = ScheduleWrapper(schedule) if schedule is not None else schedule
            self.record_shapes = record_shapes
            self.row_limit = row_limit
            self.export_to_chrome = export_to_chrome
            self.export_to_flame_graph = export_to_flame_graph
            self.profile_memory = profile_memory
            self.with_stack = export_to_flame_graph or with_stack
            self.metric = metric
            self.with_flops = with_flops
            self.local_rank = local_rank
            self.sort_by_key = sort_by_key
            self.path_to_export_trace = path_to_export_trace
            self.group_by_input_shapes = group_by_input_shapes
            self.on_trace_ready = on_trace_ready
            self.udf_on_trace_ready = True if self.on_trace_ready is not None else False
            self.record_module_names = record_module_names

            self.context_names = {}
            self.running_stack = []
            self.profiler = None
            self.lightning_module = None  # TODO: where is this set?
            self.register = None

            self.output_fname = output_filename
            self.output_file = None
            if local_rank is not None:
                super().on_train_start(local_rank=local_rank)  # TODO: why super()?
                self.on_train_start = super().on_train_start

        def _summary(self, local_rank: int) -> dict:
            self.profiler.__exit__(None, None, None)
            if self.register is not None:
                self.register.__exit__(None, None, None)

            data = self.profiler.events().key_averages(group_by_input_shapes=self.group_by_input_shapes)
            table = data.table(sort_by=self.sort_by_key, row_limit=self.row_limit)
            return {self.START_ACTION: table}

        def on_train_start(self, local_rank: Optional[str] = None, log_dir: Optional[str] = None) -> None:
            super().on_train_start(local_rank=local_rank, log_dir=log_dir)  # TODO: will this call the correct __init__?
            if self.record_module_names and self.lightning_module is not None:
                self.register = RegisterRecordFunction(self.lightning_module)
                self.register.__enter__()

        def start(self, action_name: str) -> None:
            if action_name == self.START_ACTION:
                if self.schedule is not None:
                    self.schedule._current_action = action_name
                self._create_profiler(action_name, torch.profiler.profile)

            elif action_name in self.record_functions:
                self.record_functions_managers[action_name] = record_function(action_name)
                self.record_functions_managers[action_name].__enter__()

        def stop(self, action_name: str) -> None:
            if action_name in self.record_functions:
                self.record_functions_managers[action_name].__exit__(None, None, None)

                if action_name in self.STEP_FUNCTIONS:
                    if self.schedule is not None:
                        self.schedule._current_action = action_name

                    if self.udf_on_trace_ready:
                        return

                    def on_trace_ready(profiler):
                        local_rank = 0 if self.local_rank is None else self.local_rank
                        filename = f"{action_name}_{local_rank}"
                        if self.export_to_chrome:
                            tensorboard_trace_handler(self.path_to_export_trace, filename)(profiler)
                        if self.export_to_flame_graph:
                            path = os.path.join(self.path_to_export_trace, f"{filename}.stack")
                            profiler.export_stacks(path, metric=self.metric)

                    self.profiler.on_trace_ready = on_trace_ready
                    self.profiler.step()
