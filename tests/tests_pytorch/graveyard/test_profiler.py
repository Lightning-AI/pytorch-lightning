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
import pytest


def test_v2_0_0_base_profilers():
    from pytorch_lightning.profiler.base import AbstractProfiler, BaseProfiler

    with pytest.raises(
        RuntimeError, match="AbstractProfiler` was deprecated in v1.6 and is no longer supported as of v1.9."
    ):
        AbstractProfiler()

    with pytest.raises(
        RuntimeError, match="BaseProfiler` was deprecated in v1.6 and is no longer supported as of v1.9."
    ):
        BaseProfiler()

    from pytorch_lightning.profiler.advanced import AdvancedProfiler

    with pytest.raises(
        RuntimeError, match="AdvancedProfiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        AdvancedProfiler()

    from pytorch_lightning.profiler.base import PassThroughProfiler

    with pytest.raises(
        RuntimeError, match="PassThroughProfiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        PassThroughProfiler()

    from pytorch_lightning.profiler.profiler import Profiler

    with pytest.raises(RuntimeError, match="Profiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"):
        Profiler()

    from pytorch_lightning.profiler.pytorch import PyTorchProfiler

    with pytest.raises(
        RuntimeError, match="PyTorchProfiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        PyTorchProfiler()

    from pytorch_lightning.profiler.pytorch import RegisterRecordFunction

    with pytest.raises(
        RuntimeError, match="RegisterRecordFunction` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        RegisterRecordFunction()

    from pytorch_lightning.profiler.pytorch import ScheduleWrapper

    with pytest.raises(
        RuntimeError, match="ScheduleWrapper` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        ScheduleWrapper()

    from pytorch_lightning.profiler.simple import SimpleProfiler

    with pytest.raises(
        RuntimeError, match="SimpleProfiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        SimpleProfiler()

    from pytorch_lightning.profiler.xla import XLAProfiler

    with pytest.raises(
        RuntimeError, match="XLAProfiler` was deprecated in v1.7.0 and is not longer supported as of v1.9"
    ):
        XLAProfiler()
