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

from lightning_lite.utilities.exceptions import MisconfigurationException


class _ExceptionReprMixin:
    """Mixin for custom Lightning Exceptions implementing `__repr__` method."""

    def __init_subclass__(cls) -> None:
        cls.__repr__ = _ExceptionReprMixin.__repr__

    def __repr__(self) -> str:
        str_repr = super().__repr__()
        if str_repr.startswith("_"):
            str_repr = str_repr[1:]
        return str_repr


class _ValueError(ValueError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning ValueError."""


class _RuntimeError(RuntimeError, MisconfigurationException, _ExceptionReprMixin):
    """Lighting RuntimeError."""


class _AttributeError(AttributeError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning AttributeError."""


class _TypeError(TypeError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning TypeError."""


class _NotImplementedError(NotImplementedError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning NotImplementedError."""


class _KeyError(KeyError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning KeyError."""


class _OSError(OSError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning OSError."""


class _ModuleNotFoundError(ModuleNotFoundError, MisconfigurationException, _ExceptionReprMixin):
    """Lightning ModuleNotFoundError."""


class DeadlockDetectedException(Exception):
    """Exception used when a deadlock has been detected and processes are being killed."""


class ExitGracefullyException(SystemExit):
    """Exception used when a ``signal.SIGTERM`` is sent to the process.

    This signals Lightning to try to create a fault-tolerance checkpoint once the current batch or epoch is reached
    (assuming it can be done under 30 sec). After the checkpoint is saved, Lightning will exit.
    """


class _TunerExitException(Exception):
    """Exception used to exit early while tuning."""
