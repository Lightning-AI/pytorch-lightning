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
from pytorch_lightning.profilers.pytorch import PyTorchProfiler as RedirectedPyTorchProfiler
from pytorch_lightning.profilers.pytorch import RegisterRecordFunction as RedirectedRegisterRecordFunction
from pytorch_lightning.profilers.pytorch import ScheduleWrapper as RedirectedScheduleWrapper
from pytorch_lightning.utilities import rank_zero_deprecation


class RegisterRecordFunction(RedirectedRegisterRecordFunction):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.profiler.pytorch.RegisterRecordFunction` is deprecated."
            " Use `pytorch_lightning.profilers.pytorch.RegisterRecordFunction` instead."
        )
        super().__init__(*args, **kwargs)


class ScheduleWrapper(RedirectedScheduleWrapper):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.profiler.pytorch.ScheduleWrapper` is deprecated."
            " Use `pytorch_lightning.profilers.pytorch.ScheduleWrapper` instead."
        )
        super().__init__(*args, **kwargs)


class PyTorchProfiler(RedirectedPyTorchProfiler):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.profiler.pytorch.PyTorchProfiler` is deprecated."
            " Use `pytorch_lightning.profilers.pytorch.PyTorchProfiler` instead."
        )
        super().__init__(*args, **kwargs)
