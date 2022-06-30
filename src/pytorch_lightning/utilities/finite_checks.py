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
"""Helper functions to detect NaN/Inf values."""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def print_nan_gradients(model: nn.Module) -> None:
    """Iterates over model parameters and prints out parameter + gradient information if NaN."""
    for param in model.parameters():
        if (param.grad is not None) and torch.isnan(param.grad.float()).any():
            log.info(f"{param}, {param.grad}")


def detect_nan_parameters(model: nn.Module) -> None:
    """Iterates over model parameters and prints gradients if any parameter is not finite.

    Raises:
        ValueError:
            If ``NaN`` or ``inf`` values are found
    """
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print_nan_gradients(model)
            raise ValueError(
                f"Detected nan and/or inf values in `{name}`."
                " Check your forward pass for numerically unstable operations."
            )
