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
from contextlib import redirect_stderr
from io import StringIO

from lightning_utilities.core.rank_zero import _warn, WarningCache

from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn

if __name__ == "__main__":
    stderr = StringIO()
    # recording
    with redirect_stderr(stderr):
        _warn("test1")
        _warn("test2", category=DeprecationWarning)

        rank_zero_warn("test3")
        rank_zero_warn("test4", category=DeprecationWarning)

        rank_zero_deprecation("test5")

        cache = WarningCache()
        cache.warn("test6")
        cache.deprecation("test7")

    output = stderr.getvalue()
    assert "test_warnings.py:30: UserWarning: test1" in output
    assert "test_warnings.py:31: DeprecationWarning: test2" in output

    assert "test_warnings.py:33: UserWarning: test3" in output
    assert "test_warnings.py:34: DeprecationWarning: test4" in output

    assert "test_warnings.py:36: LightningDeprecationWarning: test5" in output

    assert "test_warnings.py:39: UserWarning: test6" in output
    assert "test_warnings.py:40: LightningDeprecationWarning: test7" in output

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
