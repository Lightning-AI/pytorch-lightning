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

import sys
from typing import Any


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["pytorch_lightning.utilities.memory"] = self


def get_gpu_memory_map() -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`pl.utilities.memory.get_gpu_memory_map` was deprecated in v1.5 and is no longer supported as of v1.9."
        " Use `pl.accelerators.cuda.get_nvidia_gpu_stats` instead."
    )


_patch_sys_modules()
