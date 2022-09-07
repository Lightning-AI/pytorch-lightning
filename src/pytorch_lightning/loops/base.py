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
from pytorch_lightning.loops import Loop as NewLoop
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class Loop(NewLoop):
    def __init__(self) -> None:
        rank_zero_deprecation(
            "pytorch_lightning.loops.base.Loop has been deprecated in v1.7"
            " and will be removed in v1.9."
            " Use the equivalent class from the pytorch_lightning.loops.loop.Loop class instead."
        )
        super().__init__()
