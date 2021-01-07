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
from typing import Tuple, Optional

import torch
from pytorch_lightning.metrics.utils import to_categorical, reduce


def _dice_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    bg: bool = False,
) -> [torch.Tensor, torch.Tensor, torch.Tensor]:

    num_classes = preds.shape[1]
    bg = 1 - int(bool(bg))
    tps = torch.zeros(num_classes - bg, device=preds.device, dtype=torch.float32)
    fps = torch.zeros(num_classes - bg, device=preds.device, dtype=torch.float32)
    fns = torch.zeros(num_classes - bg, device=preds.device, dtype=torch.float32)

    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=1)
    # TODO: should use _input_format_classification() here?

    for c in range(bg, num_classes):
        tps[c] = torch.sum((preds == c) * (target == c))
        fps[c] = torch.sum((preds == c) * (target != c))
        fns[c] = torch.sum((preds != c) * (target == c))

    return tps, fps, fns


def _dice_compute(
    tps: torch.Tensor, fps: torch.Tensor, fns: torch.Tensor, reduction: str = "elementwise_mean", nan_score: float = 0.0
) -> torch.Tensor:
    num = 2 * tps
    denom = 2 * tps + fps + fns
    scores = num / denom
    scores = torch.nan_to_num(scores, nan_score)
    return reduce(scores, reduction=reduction)


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> torch.Tensor:
    """
    Compute dice score from prediction scores

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor containing dice score

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> dice_score(pred, target)
        tensor(0.3333)

    """
    tps, fps, fns = _dice_update(pred, target, bg)
    return _dice_compute(tps, fps, fns, reduction, nan_score)
