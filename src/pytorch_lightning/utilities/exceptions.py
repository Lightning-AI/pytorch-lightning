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

from lightning_fabric.utilities.exceptions import MisconfigurationException  # noqa: F401


class DeadlockDetectedException(Exception):
    """Exception used when a deadlock has been detected and processes are being killed."""


class ExitGracefullyException(SystemExit):
    """Exception used when a ``signal.SIGTERM`` is sent to the process.

    This signals Lightning to try to create a fault-tolerance checkpoint once the current batch or epoch is reached
    (assuming it can be done under 30 sec). After the checkpoint is saved, Lightning will exit.
    """


class _TunerExitException(Exception):
    """Exception used to exit early while tuning."""
