# Copyright The Lightning AI team.
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

from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.utilities.types import Optimizable


class XLAPrecision(Precision):
    """Precision plugin with XLA."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        import torch_xla.core.xla_model as xm

        # you always want to `xm.mark_step()` after `optimizer.step` for better performance, so we set `barrier=True`
        return xm.optimizer_step(optimizer, optimizer_args=kwargs, barrier=True)
