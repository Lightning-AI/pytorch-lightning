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


class ProfilerConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, profiler: Union[BaseProfiler, bool, str]):

        if isinstance(profiler, bool):
            rank_zero_warn("Passing a bool value as a `profiler` argument to Trainer is deprecated"
                           " and will be removed in v1.2. Use str ('simple' or 'advanced') instead.",
                           DeprecationWarning)
        # configure profiler
        if profiler is True or profiler == "simple":
            profiler = SimpleProfiler()
        elif profiler == "advanced":
            profiler = AdvancedProfiler()
        elif isinstance(profiler, str):
            raise ValueError("When passing string value for the `profiler` parameter of"
                             " `Trainer`, it can only be 'simple' or 'advanced'")
        self.trainer.profiler = profiler or PassThroughProfiler()
