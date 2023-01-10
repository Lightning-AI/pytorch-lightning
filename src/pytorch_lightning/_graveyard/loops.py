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
    sys.modules["pytorch_lightning.loops.base"] = self


class Loop:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.loops.base.Loop` was deprecated in v1.7.0 and removed as of v1.9.0."
            " Please use `pytorch_lightning.loops.loop.Loop` instead"
        )


_patch_sys_modules()
