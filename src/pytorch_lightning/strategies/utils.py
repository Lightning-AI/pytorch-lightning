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
import importlib
import os
from inspect import getmembers, isclass

import torch
from torch import Tensor
from typing_extensions import Literal

from lightning_fabric.plugins.precision.utils import _convert_fp_tensor
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities.registry import _is_register_method_overridden
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


def on_colab_kaggle() -> bool:
    rank_zero_deprecation("The function `on_colab_kaggle` has been deprecated in v1.8.0 and will be removed in v2.0.0.")
    return bool(os.getenv("COLAB_GPU") or os.getenv("KAGGLE_URL_BASE"))


def _call_register_strategies(registry: _StrategyRegistry, base_module: str) -> None:
    # TODO(fabric): Remove this function once PL strategies inherit from Fabrics Strategy base class
    module = importlib.import_module(base_module)
    for _, mod in getmembers(module, isclass):
        if issubclass(mod, Strategy) and _is_register_method_overridden(mod, Strategy, "register_strategies"):
            mod.register_strategies(registry)


def _fp_to_half(tensor: Tensor, precision: Literal["64", 64, "32", 32, "16", 16, "bf16"]) -> Tensor:
    if str(precision) == "16":
        return _convert_fp_tensor(tensor, torch.half)
    if precision == "bf16":
        return _convert_fp_tensor(tensor, torch.bfloat16)
    return tensor
