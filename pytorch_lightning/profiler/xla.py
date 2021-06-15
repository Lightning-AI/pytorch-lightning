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
"""Profiler to check if there are any bottlenecks in your code."""
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from pytorch_lightning.profiler.base import BaseProfiler
from pytorch_lightning.utilities import _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.debug.profiler as xp

log = logging.getLogger(__name__)


class XLAProfiler(BaseProfiler):

    STEP_FUNCTIONS = {
        "training_step_and_backward",
        "validation_step",
        "test_step",
        "predict_step",
    }

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ):
        super().__init__(dirpath=dirpath, filename=filename, output_filename=output_filename)
        self._recording_map: Dict = {}

    def start(self, action_name: str) -> None:
        recording = xp.Trace(action_name)
        recording.__enter__()
        self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

    def summary(self) -> str:
        return ""
