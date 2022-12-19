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
    sys.modules["pytorch_lightning.profiler.advanced"] = self
    sys.modules["pytorch_lightning.profiler.base"] = self
    sys.modules["pytorch_lightning.profiler.profiler"] = self
    sys.modules["pytorch_lightning.profiler.pytorch"] = self
    sys.modules["pytorch_lightning.profiler.simple"] = self
    sys.modules["pytorch_lightning.profiler.xla"] = self


class AbstractProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.profiler.base.AbstractProfiler` was deprecated in v1.6 and is no longer supported"
            " as of v1.9. Use `pytorch_lightning.profilers.Profiler` instead."
        )


class BaseProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.base.BaseProfiler` was deprecated in v1.6 and is no longer supported"
            " as of v1.9. Use `pytorch_lightning.profilers.Profiler` instead."
        )


class AdvancedProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.advanced.AdvancedProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.AdvancedProfiler` instead."
        )


class PassThroughProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.base.PassThroughProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.PassThroughProfiler` instead."
        )


class Profiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.profiler.Profiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.Profiler` instead."
        )


class PyTorchProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.pytorch.PyTorchProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.PyTorchProfiler` instead."
        )


class RegisterRecordFunction:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.pytorch.RegisterRecordFunction` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.pytorch.RegisterRecordFunction` instead."
        )


class ScheduleWrapper:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.pytorch.ScheduleWrapper` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.pytorch.ScheduleWrapper` instead."
        )


class SimpleProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.simple.SimpleProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.SimpleProfiler` instead."
        )


class XLAProfiler:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "`pytorch_lightning.profiler.xla.XLAProfiler` was deprecated in v1.7.0 and is not longer"
            " supported as of v1.9.0. Use `pytorch_lightning.profilers.XLAProfiler` instead."
        )


_patch_sys_modules()
