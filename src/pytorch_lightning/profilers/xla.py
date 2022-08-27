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
import logging
from typing import Dict

from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TPU_AVAILABLE:
    import torch_xla.debug.profiler as xp

log = logging.getLogger(__name__)


class XLAProfiler(Profiler):

    STEP_FUNCTIONS = {"validation_step", "test_step", "predict_step"}
    RECORD_FUNCTIONS = {
        "training_step",
        "backward",
        "validation_step",
        "test_step",
        "predict_step",
    }

    def __init__(self, port: int = 9012) -> None:
        """XLA Profiler will help you debug and optimize training workload performance for your models using Cloud
        TPU performance tools.

        Args:
            port: the port to start the profiler server on. An exception is
                raised if the provided port is invalid or busy.
        """
        if not _TPU_AVAILABLE:
            raise MisconfigurationException("`XLAProfiler` is only supported on TPUs")
        super().__init__(dirpath=None, filename=None)
        self.port = port
        self._recording_map: Dict = {}
        self._step_recoding_map: Dict = {}
        self._start_trace: bool = False

    def start(self, action_name: str) -> None:
        if action_name in self.RECORD_FUNCTIONS:
            if not self._start_trace:
                self.server = xp.start_server(self.port)
                self._start_trace = True

            if action_name in self.STEP_FUNCTIONS:
                step = self._get_step_num(action_name)
                recording = xp.StepTrace(action_name, step_num=step)
            else:
                recording = xp.Trace(action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

    def _get_step_num(self, action_name: str) -> int:
        if action_name not in self._step_recoding_map:
            self._step_recoding_map[action_name] = 1
        else:
            self._step_recoding_map[action_name] += 1
        return self._step_recoding_map[action_name]
