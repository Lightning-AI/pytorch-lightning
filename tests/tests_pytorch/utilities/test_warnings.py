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
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

import lightning.pytorch
import pytest
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from lightning_utilities.test.warning import no_warning_call

if __name__ == "__main__":
    # check that logging is properly configured
    import logging

    root_logger = logging.getLogger()
    lightning_logger = logging.getLogger("lightning.pytorch")
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


@pytest.mark.parametrize("setting", ["0", "off"])
@mock.patch.dict(os.environ, {}, clear=True)
def test_disable_possible_user_warnings_from_environment(setting):
    with pytest.warns(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    os.environ["POSSIBLE_USER_WARNINGS"] = setting
    importlib.reload(lightning.pytorch)
    with no_warning_call(PossibleUserWarning):
        warnings.warn("test", PossibleUserWarning)
    warnings.resetwarnings()
