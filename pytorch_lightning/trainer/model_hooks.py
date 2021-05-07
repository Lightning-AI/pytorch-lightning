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

from abc import ABC
from typing import Optional

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_deprecation
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature


class TrainerModelHooksMixin(ABC):
    """
    TODO: Remove this class in v1.6.

    Use the utilities from ``pytorch_lightning.utilities.signature_utils`` instead.
    """

    lightning_module: LightningModule

    def is_function_implemented(self, f_name: str, model: Optional[LightningModule] = None) -> bool:
        rank_zero_deprecation(
            "Internal: TrainerModelHooksMixin.is_function_implemented is deprecated in v1.4"
            " and will be removed in v1.6."
        )
        # note: currently unused - kept as it is public
        if model is None:
            model = self.lightning_module
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def has_arg(self, f_name: str, arg_name: str) -> bool:
        rank_zero_deprecation(
            "Internal: TrainerModelHooksMixin.is_function_implemented is deprecated in v1.4"
            " and will be removed in v1.6."
            " Use `pytorch_lightning.utilities.signature_utils.is_param_in_hook_signature` instead."
        )
        model = self.lightning_module
        f_op = getattr(model, f_name, None)
        if not f_op:
            return False
        return is_param_in_hook_signature(f_op, arg_name)
