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
from typing import Any

from pytorch_lightning.profilers.advanced import AdvancedProfiler as NewAdvancedProfiler
from pytorch_lightning.profilers.base import PassThroughProfiler as NewPassThroughProfiler
from pytorch_lightning.profilers.profiler import Profiler as NewProfiler
from pytorch_lightning.profilers.pytorch import PyTorchProfiler as NewPyTorchProfiler
from pytorch_lightning.profilers.simple import SimpleProfiler as NewSimpleProfiler
from pytorch_lightning.profilers.xla import XLAProfiler as NewXLAProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class AdvancedProfiler(NewAdvancedProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.AdvancedProfiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.AdvancedProfiler` class instead."
        )
        super().__init__(*args, **kwargs)


class PassThroughProfiler(NewPassThroughProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.PassThroughProfiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.PassThroughProfiler` class instead."
        )
        super().__init__(*args, **kwargs)


class Profiler(NewProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.Profiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.Profiler` class instead."
        )
        super().__init__(*args, **kwargs)


class PyTorchProfiler(NewPyTorchProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.PyTorchProfiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.PyTorchProfiler` class instead."
        )
        super().__init__(*args, **kwargs)


class SimpleProfiler(NewSimpleProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.SimpleProfiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.SimpleProfiler` class instead."
        )
        super().__init__(*args, **kwargs)


class XLAProfiler(NewXLAProfiler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.profiler.XLAProfiler` is deprecated in v1.9.0 and will be removed in v2.0.0."
            " Use the equivalent `pytorch_lightning.profilers.XLAProfiler` class instead."
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "Profiler",
    "AdvancedProfiler",
    "PassThroughProfiler",
    "PyTorchProfiler",
    "SimpleProfiler",
    "XLAProfiler",
]
