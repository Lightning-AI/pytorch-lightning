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

import subprocess
import sys
from queue import Empty
from typing import List, Optional, Tuple

from packaging.version import Version

from lightning_app import LightningFlow, LightningWork
from lightning_app.core.queues import BaseQueue
from lightning_app.utilities.imports import (
    _CLOUD_TEST_RUN,
    _is_lightning_flash_available,
    _is_pytorch_lightning_available,
)


def _call_script(
    filepath: str,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = 60 * 10,
) -> Tuple[int, str, str]:
    if args is None:
        args = []
    args = [str(a) for a in args]
    command = [sys.executable, filepath] + args  # todo: add back coverage
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return p.returncode, stdout, stderr


def _run_script(filepath):
    code, stdout, stderr = _call_script(filepath)
    print(f"{filepath} STDOUT: {stdout}")
    print(f"{filepath} STDERR: {stderr}")
    assert not code, code


class _RunIf:
    """RunIf wrapper for simple marking specific cases, fully compatible with pytest.mark::

    @RunIf(...)
    @pytest.mark.parametrize("arg1", [1, 2.0])
    def test_wrapper(arg1):
        assert arg1 > 0.0
    """

    def __new__(
        self,
        *args,
        pl: bool = False,
        flash: bool = False,
        min_python: Optional[str] = None,
        skip_windows: bool = False,
        skip_linux: bool = False,
        skip_mac_os: bool = False,
        local_end_to_end: bool = False,
        cloud: bool = False,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` arguments.
            pl: Requires that PyTorch Lightning is installed.
            flash: Requires that Flash is installed.
            min_python: Require that Python is greater or equal than this version.
            skip_windows: Skip for Windows platform.
            skip_linux: Skip for Linux platform.
            skip_mac_os: Skip for Mac Os Platform.
            **kwargs: Any :class:`pytest.mark.skipif` keyword arguments.
        """
        import pytest

        conditions = []
        reasons = []

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if skip_linux:
            conditions.append(sys.platform == "linux")
            reasons.append("unimplemented on linux")

        if skip_mac_os:
            conditions.append(sys.platform == "darwin")
            reasons.append("unimplemented on MacOS")

        if pl:
            conditions.append(not _is_pytorch_lightning_available())
            reasons.append("PyTorch Lightning is required.")

        if flash:
            conditions.append(not _is_lightning_flash_available())
            reasons.append("Lightning Flash is required.")

        if cloud:
            conditions.append(not _CLOUD_TEST_RUN)
            reasons.append("Cloud End to End tests should not run.")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args, condition=any(conditions), reason=f"Requires: [{' + '.join(reasons)}]", **kwargs
        )


class _MockQueue(BaseQueue):
    def __init__(self, name: str = "", default_timeout: float = 0):
        super().__init__(name, default_timeout)
        self._queue = []

    def put(self, item):
        self._queue.append(item)

    def get(self, timeout: int = 0):
        if not self._queue:
            raise Empty()
        return self._queue.pop(0)

    def __contains__(self, item):
        return item in self._queue

    def __len__(self):
        return len(self._queue)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._queue})"


class EmptyFlow(LightningFlow):
    """A LightningFlow that implements all abstract methods to do nothing.

    Useful for mocking in tests.
    """

    def run(self):
        pass


class EmptyWork(LightningWork):
    """A LightningWork that implements all abstract methods to do nothing.

    Useful for mocking in tests.
    """

    def run(self):
        pass
