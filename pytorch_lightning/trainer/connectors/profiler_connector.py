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
# limitations under the License

from typing import Union

from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler, SimpleProfiler, AdvancedProfiler
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

PROFILERS = {
    "simple": SimpleProfiler,
    "advanced": AdvancedProfiler,
}


class ProfilerConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, profiler: Union[BaseProfiler, bool, str]):

        if profiler and not isinstance(profiler, (bool, str, BaseProfiler)):
            # TODO: Update exception on removal of bool
            raise MisconfigurationException("Only None, bool, str and subclasses of `BaseProfiler`"
                                            " are valid values for `Trainer`'s `profiler` parameter."
                                            f" Received {profiler} which is of type {type(profiler)}.")

        if isinstance(profiler, bool):
            rank_zero_warn("Passing a bool value as a `profiler` argument to `Trainer` is deprecated"
                           " and will be removed in v1.3. Use str ('simple' or 'advanced') instead.",
                           DeprecationWarning)
            if profiler:
                profiler = SimpleProfiler()
        elif isinstance(profiler, str):
            if profiler.lower() in PROFILERS:
                profiler_class = PROFILERS[profiler.lower()]
                profiler = profiler_class()
            else:
                raise ValueError("When passing string value for the `profiler` parameter of"
                                 " `Trainer`, it can only be 'simple' or 'advanced'")
        self.trainer.profiler = profiler or PassThroughProfiler()
