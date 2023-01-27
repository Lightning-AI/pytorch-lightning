# Copyright The Lightning team.
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
from lightning_fabric.utilities.exceptions import MisconfigurationException  # noqa: F401


class SIGTERMException(SystemExit):
    """Exception used when a :class:`signal.SIGTERM` is sent to a process.

    This exception is raised by the loops at specific points. It can be used to write custom logic in the
    :meth:`pytorch_lightning.callbacks.callback.Callback.on_exception` method.

    For example, you could use the :class:`pytorch_lightning.callbacks.fault_tolerance.OnExceptionCheckpoint` callback
    that saves a checkpoint for you when this exception is raised.
    """


class _TunerExitException(Exception):
    """Exception used to exit early while tuning."""
