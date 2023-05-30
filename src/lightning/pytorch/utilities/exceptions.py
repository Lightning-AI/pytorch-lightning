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
import re

from lightning.fabric.utilities.exceptions import MisconfigurationException  # noqa: F401
from lightning.pytorch.utilities.imports import _PYTHON_GREATER_EQUAL_3_11_0


class SIGTERMException(SystemExit):
    """Exception used when a :class:`signal.SIGTERM` is sent to a process.

    This exception is raised by the loops at specific points. It can be used to write custom logic in the
    :meth:`lightning.pytorch.callbacks.callback.Callback.on_exception` method.

    For example, you could use the :class:`lightning.pytorch.callbacks.fault_tolerance.OnExceptionCheckpoint` callback
    that saves a checkpoint for you when this exception is raised.
    """


class _TunerExitException(Exception):
    """Exception used to exit early while tuning."""


def _augment_message(exception: BaseException, pattern: str, new_message: str) -> None:
    if _PYTHON_GREATER_EQUAL_3_11_0 and any(re.match(pattern, message, re.DOTALL) for message in exception.args):
        exception.add_note(new_message)
    else:
        # Remove this when Python 3.11 becomes the minimum supported version
        exception.args = tuple(
            new_message if re.match(pattern, message, re.DOTALL) else message for message in exception.args
        )
