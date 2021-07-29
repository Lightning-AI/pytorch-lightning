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
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.utilities import rank_zero_deprecation


class DeprecatedTrainerAttributes:

    sanity_checking: bool
    fit_loop: FitLoop

    @property
    def running_sanity_check(self) -> bool:
        rank_zero_deprecation(
            "`Trainer.running_sanity_check` has been renamed to `Trainer.sanity_checking` and will be removed in v1.5."
        )
        return self.sanity_checking

    @property
    def train_loop(self) -> FitLoop:
        rank_zero_deprecation(
            "`Trainer.train_loop` has been renamed to `Trainer.fit_loop` and will be removed in v1.6."
        )
        return self.fit_loop
