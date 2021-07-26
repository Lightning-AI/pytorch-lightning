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
"""
XLA Profiler will help you debug and optimize training workload performance
for your models using Cloud TPU performance tools.

Manual capture via TensorBoard

The following instructions are for capturing trace from a running program

0. This [guide](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm#tpu-vm) will
help you with the Cloud TPU setup with the required installations

1. Start a TensorBoard Server

>> tensorboard --logdir ./tensorboard --port 9001

You could view the TensorBoard output at http://localhost:9001 on your local machine, and then open the
``PROFILE`` plugin from the top right dropdown or open http://localhost:9001/#profile

2. Once the code you'd like to profile is running, click on ``CAPTURE PROFILE`` button. You could enter
``localhost:9012`` (default port for XLA Profiler) as the Profile Service URL. Then, you could enter
the number of milliseconds for the profiling duration, and click ``CAPTURE``

3. Make sure the code is running, while you are trying to capture the traces. Also, it would lead to better
performance insights if the profiling duration is longer than the step time

4. Once the capture is finished, the page will refresh and you could browse through the insights using the
``Tools`` dropdown at the top left

"""
import logging
from typing import Dict

from pytorch_lightning.profiler.base import BaseProfiler
from pytorch_lightning.utilities import _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.debug.profiler as xp

log = logging.getLogger(__name__)


class XLAProfiler(BaseProfiler):

    STEP_FUNCTIONS = {"training_step_and_backward", "validation_step", "test_step", "predict_step"}
    RECORD_FUNCTIONS = {
        "training_step_and_backward",
        "training_step",
        "backward",
        "validation_step",
        "test_step",
        "predict_step",
    }

    def __init__(self, port: int = 9012) -> None:
        """
        This Profiler will help you debug and optimize training workload performance
        for your models using Cloud TPU performance tools.
        """
        super().__init__(dirpath=None, filename=None, output_filename=None)
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

    def summary(self) -> str:
        return ""
