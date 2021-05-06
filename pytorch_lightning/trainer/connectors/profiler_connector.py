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
from weakref import proxy

from pytorch_lightning.profiler import (
    AdvancedProfiler,
    BaseProfiler,
    PassThroughProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException

PROFILERS = {
    "simple": SimpleProfiler,
    "advanced": AdvancedProfiler,
    "pytorch": PyTorchProfiler,
}


class ProfilerConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, profiler: Union[BaseProfiler, str]):

        if profiler and not isinstance(profiler, (str, BaseProfiler)):
            raise MisconfigurationException(
                "Only None, str and subclasses of `BaseProfiler`"
                " are valid values for `Trainer`'s `profiler` parameter."
                f" Received {profiler} which is of type {type(profiler)}."
            )
        if isinstance(profiler, str):
            if profiler.lower() in PROFILERS:
                profiler_class = PROFILERS[profiler.lower()]
                profiler = profiler_class()
            else:
                raise ValueError(
                    "When passing string value for the `profiler` parameter of"
                    " `Trainer`, it can only be 'simple', 'advanced' or 'pytorch'"
                )
        self.trainer.profiler = profiler or PassThroughProfiler()

    def setup(self) -> None:
        trainer = self.trainer
        local_rank = trainer.local_rank if trainer.world_size > 1 else None
        trainer.profiler._lightning_module = proxy(trainer.lightning_module)
        trainer.profiler.setup(stage=trainer.state.fn._setup_fn, local_rank=local_rank, log_dir=trainer.log_dir)
