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

from typing import Any


class DDP2Strategy:
    """DDP2 behaves like DP in one node, but synchronization across nodes behaves like in DDP.

    .. deprecated:: v1.7
        This strategy is no longer supported in v1.7 will be removed completely in v1.8. For single-node execution, we
        recommend the :class:`~pytorch_lightning.strategies.ddp.DDPStrategy` or the
        :class:`~pytorch_lightning.strategies.dp.DataParallelStrategy` as a replacement. If you rely on DDP2, you will
        need ``torch < 1.9`` and ``pytorch-lightning < 1.5``.
    """

    strategy_name = "ddp2"

    def __new__(cls, *args: Any, **kwargs: Any) -> "DDP2Strategy":
        raise TypeError(
            "The `DDP2Strategy`/`DDP2Plugin` is no longer supported in v1.7 and will be removed completely in v1.8."
            " For single-node execution, we recommend the `DDPStrategy` or the `DPStrategy`. If you rely on DDP2, you"
            " will need `torch < 1.9` and `pytorch-lightning < 1.5`."
        )
