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
from typing import Tuple, Union

import torch
from pytorch_lightning.metrics.classification.helpers import _input_format_classification


def _hamming_loss_update(preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, int]:
    preds, target, _ = _input_format_classification(preds, target, threshold=threshold)

    correct = (preds == target).sum()
    total = preds.numel()

    return correct, total


def _hamming_loss_compute(correct: torch.Tensor, total: Union[int, torch.Tensor]) -> torch.Tensor:
    return 1 - correct.float() / total


def hamming_loss(preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    r"""
    Computes the average Hamming loss or `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_
    between targets and predictions:

    .. math::
        \text{Hamming loss} = \frac{1}{N \cdot L} \sum_i^N \sum_l^L 1(y_{il} \neq \hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label. If this is not what you want, consider using
    :class:`~pytorch_lightning.metrics.classification.Accuracy`.

    Accepts all input types listed in :ref:`metrics:Input types`.

    Args:
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5

    Example:

        >>> from pytorch_lightning.metrics.functional import hamming_loss
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> hamming_loss(preds, target)
        tensor(0.2500)

    """
    assert 0 <= threshold <= 1, f"threshold: {threshold} is out of range"
    correct, total = _hamming_loss_update(preds, target, threshold)
    return _hamming_loss_compute(correct, total)
