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
"""Test that the warnings actually appear and they have the correct `stacklevel`

Needs to be run outside of `pytest` as it captures all the warnings.

"""

import importlib
import inspect
import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

import lightning.fabric
import pytest
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning.fabric.utilities.warnings import (
    PossibleUserWarning,
    _is_path_in_lightning,
    disable_possible_user_warnings,
)
from lightning_utilities.core.rank_zero import WarningCache, _warn
from lightning_utilities.test.warning import no_warning_call


def line_number():
    return inspect.currentframe().f_back.f_lineno


if __name__ == "__main__":
    stderr = StringIO()
    # recording
    with redirect_stderr(stderr):
        base_line = line_number() + 1
        _warn("test1")
        _warn("test2", category=DeprecationWarning)

        rank_zero_warn("test3")
        rank_zero_warn("test4", category=DeprecationWarning)

        rank_zero_deprecation("test5")

        cache = WarningCache()
        cache.warn("test6")
        cache.deprecation("test7")

    output = stderr.getvalue()
    expected_lines = [
        f"test_warnings.py:{base_line}: test1",
        f"test_warnings.py:{base_line + 1}: test2",
        f"test_warnings.py:{base_line + 3}: test3",
        f"test_warnings.py:{base_line + 4}: test4",
        f"test_warnings.py:{base_line + 6}: test5",
        f"test_warnings.py:{base_line + 9}: test6",
        f"test_warnings.py:{base_line + 10}: test7",
    ]

    for ln in expected_lines:
        assert ln in output, f"Missing line {ln!r} in:\n{output}"
        assert "rank_zero_warn(" not in output, "customized warning wrapper is not used"

    # check that logging is properly configured
    import logging

    root_logger = logging.getLogger()
    lightning_logger = logging.getLogger("lightning.fabric")
    # should have a `StreamHandler`
    assert lightning_logger.hasHandlers()
    assert len(lightning_logger.handlers) == 1
    # set our own stream for testing
    handler = lightning_logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    stderr = StringIO()
    # necessary with `propagate = False`
    lightning_logger.handlers[0].stream = stderr

    # necessary with `propagate = True`
    with redirect_stderr(stderr):
        # Lightning should not configure the root `logging` logger by default
        logging.info("test1")
        root_logger.info("test1")
        # but our logger instance
        lightning_logger.info("test2")
        # level is set to INFO
        lightning_logger.debug("test3")

    output = stderr.getvalue()
    assert output == "test2\n", repr(output)


def test_is_path_in_lightning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if sys.platform != "win32":
        assert _is_path_in_lightning(Path("/a/b/c/lightning"))
        assert _is_path_in_lightning(Path("/a/b/c/lightning/core/lightning.py"))
        assert _is_path_in_lightning(Path("/a/b/c/lightning/lightning"))
        assert not _is_path_in_lightning(Path("/a/b/c/"))
        # The following statements should assert the opposite for correctness, but a naive implementation of
        # `_is_path_in_lightning` was requested, thus it cannot handle these cases
        assert _is_path_in_lightning(Path(""))
        assert _is_path_in_lightning(Path("/a/b/lightning"))
        assert _is_path_in_lightning(Path("a/b/c/lightning"))
    else:
        assert _is_path_in_lightning(Path(r"C:\a\b\c\lightning"))
        assert _is_path_in_lightning(Path(r"C:\a\b\c\lightning\core\lightning.py"))
        assert _is_path_in_lightning(Path(r"C:\a\b\c\lightning\lightning"))
        assert not _is_path_in_lightning(Path(r"\a\b\c"))
        assert not _is_path_in_lightning(Path(r"C:\a\b\c"))
        # The following statements should assert the opposite for correctness, but a naive implementation of
        # `_is_path_in_lightning` was requested, thus it cannot handle these cases
        assert _is_path_in_lightning(Path(r"D:\a\b\c\lightning"))  # drive letter mismatch


def test_disable_possible_user_warnings():
    with pytest.warns(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    disable_possible_user_warnings()
    with no_warning_call(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    warnings.resetwarnings()


@pytest.mark.parametrize("setting", ["0", "off"])
@mock.patch.dict(os.environ, {}, clear=True)
def test_disable_possible_user_warnings_from_environment(setting):
    with pytest.warns(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    os.environ["POSSIBLE_USER_WARNINGS"] = setting
    importlib.reload(lightning.fabric)

    with no_warning_call(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    warnings.resetwarnings()
