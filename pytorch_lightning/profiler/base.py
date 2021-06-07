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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Union

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem

log = logging.getLogger(__name__)


class AbstractProfiler(ABC):
    """Specification of a profiler."""

    @abstractmethod
    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""

    @abstractmethod
    def stop(self, action_name: str) -> None:
        """Defines how to record the duration once an action is complete."""

    @abstractmethod
    def summary(self) -> str:
        """Create profiler summary in text format."""

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """Execute arbitrary pre-profiling set-up steps as defined by subclass."""

    @abstractmethod
    def teardown(self, **kwargs: Any) -> None:
        """Execute arbitrary post-profiling tear-down steps as defined by subclass."""


class BaseProfiler(AbstractProfiler):
    """
    If you wish to write a custom profiler, you should inherit from this class.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> None:
        self.dirpath = dirpath
        self.filename = filename
        if output_filename is not None:
            rank_zero_warn(
                "`Profiler` signature has changed in v1.3. The `output_filename` parameter has been removed in"
                " favor of `dirpath` and `filename`. Support for the old signature will be removed in v1.5",
                DeprecationWarning
            )
            filepath = Path(output_filename)
            self.dirpath = filepath.parent
            self.filename = filepath.stem

        self._output_file: Optional[TextIO] = None
        self._write_stream: Optional[Callable] = None
        self._local_rank: Optional[int] = None
        self._log_dir: Optional[str] = None
        self._stage: Optional[str] = None

    @contextmanager
    def profile(self, action_name: str) -> None:
        """
        Yields a context manager to encapsulate the scope of a profiled action.

        Example::

            with self.profile('load training data'):
                # load training data code

        The profiler will start once you've entered the context and will automatically
        stop once you exit the code block.
        """
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def profile_iterable(self, iterable, action_name: str) -> None:
        iterator = iter(iterable)
        while True:
            try:
                self.start(action_name)
                value = next(iterator)
                self.stop(action_name)
                yield value
            except StopIteration:
                self.stop(action_name)
                break

    def _rank_zero_info(self, *args, **kwargs) -> None:
        if self._local_rank in (None, 0):
            log.info(*args, **kwargs)

    def _prepare_filename(self, extension: str = ".txt") -> str:
        filename = ""
        if self._stage is not None:
            filename += f"{self._stage}-"
        filename += str(self.filename)
        if self._local_rank is not None:
            filename += f"-{self._local_rank}"
        filename += extension
        return filename

    def _prepare_streams(self) -> None:
        if self._write_stream is not None:
            return
        if self.filename:
            filepath = os.path.join(self.dirpath, self._prepare_filename())
            fs = get_filesystem(filepath)
            file = fs.open(filepath, "a")
            self._output_file = file
            self._write_stream = file.write
        else:
            self._write_stream = self._rank_zero_info

    def describe(self) -> None:
        """Logs a profile report after the conclusion of run."""
        # there are pickling issues with open file handles in Python 3.6
        # so to avoid them, we open and close the files within this function
        # by calling `_prepare_streams` and `teardown`
        self._prepare_streams()
        summary = self.summary()
        if summary:
            self._write_stream(summary)
        if self._output_file is not None:
            self._output_file.flush()
        self.teardown(stage=self._stage)

    def _stats_to_str(self, stats: Dict[str, str]) -> str:
        stage = f"{self._stage.upper()} " if self._stage is not None else ""
        output = [stage + "Profiler Report"]
        for action, value in stats.items():
            header = f"Profile stats for: {action}"
            if self._local_rank is not None:
                header += f" rank: {self._local_rank}"
            output.append(header)
            output.append(value)
        return os.linesep.join(output)

    def setup(
        self,
        stage: Optional[str] = None,
        local_rank: Optional[int] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        """Execute arbitrary pre-profiling set-up steps."""
        self._stage = stage
        self._local_rank = local_rank
        self._log_dir = log_dir
        self.dirpath = self.dirpath or log_dir

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Execute arbitrary post-profiling tear-down steps.

        Closes the currently open file and stream.
        """
        self._write_stream = None
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None  # can't pickle TextIOWrapper

    def __del__(self) -> None:
        self.teardown(stage=self._stage)

    def start(self, action_name: str) -> None:
        raise NotImplementedError

    def stop(self, action_name: str) -> None:
        raise NotImplementedError

    def summary(self) -> str:
        raise NotImplementedError

    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank


class PassThroughProfiler(BaseProfiler):
    """
    This class should be used when you don't want the (small) overhead of profiling.
    The Trainer uses this class by default.
    """

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    def summary(self) -> str:
        return ""
