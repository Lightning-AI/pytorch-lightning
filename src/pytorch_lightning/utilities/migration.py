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
from __future__ import annotations

import sys
import threading
from types import ModuleType, TracebackType

import pytorch_lightning.utilities.argparse

# Create a global lock to ensure no race condition with deleting sys modules
_lock = threading.Lock()


class pl_legacy_patch:
    """Registers legacy artifacts (classes, methods, etc.) that were removed but still need to be included for
    unpickling old checkpoints. The following patches apply.

        1. ``pytorch_lightning.utilities.argparse._gpus_arg_default``: Applies to all checkpoints saved prior to
           version 1.2.8. See: https://github.com/Lightning-AI/lightning/pull/6898
        2. ``pytorch_lightning.utilities.argparse_utils``: A module that was deprecated in 1.2 and removed in 1.4,
           but still needs to be available for import for legacy checkpoints.

    Example:

        with pl_legacy_patch():
            torch.load("path/to/legacy/checkpoint.ckpt")
    """

    def __enter__(self) -> None:
        _lock.acquire()
        # `pl.utilities.argparse_utils` was renamed to `pl.utilities.argparse`
        legacy_argparse_module = ModuleType("pytorch_lightning.utilities.argparse_utils")
        sys.modules["pytorch_lightning.utilities.argparse_utils"] = legacy_argparse_module

        # `_gpus_arg_default` used to be imported from these locations
        legacy_argparse_module._gpus_arg_default = lambda x: x
        pytorch_lightning.utilities.argparse._gpus_arg_default = lambda x: x

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> None:
        if hasattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default"):
            delattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
        del sys.modules["pytorch_lightning.utilities.argparse_utils"]
        _lock.release()
