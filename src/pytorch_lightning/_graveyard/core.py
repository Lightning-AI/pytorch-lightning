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

from pytorch_lightning import LightningModule


def _use_amp(_: LightningModule) -> None:
    # Remove in v2.0.0
    raise AttributeError(
        "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


def _use_amp_setter(_: LightningModule, __: bool) -> None:
    # Remove in v2.0.0
    raise AttributeError(
        "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


LightningModule.use_amp = property(fget=_use_amp, fset=_use_amp_setter)
