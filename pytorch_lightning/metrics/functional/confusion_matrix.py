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
from typing import Optional

import torch

from pytorch_lightning.metrics.classification.helpers import _input_format_classification, DataType
from pytorch_lightning.utilities import rank_zero_warn


def _confusion_matrix_update(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, threshold: float = 0.5
) -> torch.Tensor:
    preds, target, mode = _input_format_classification(preds, target, threshold)
    if mode not in (DataType.BINARY, DataType.MULTILABEL):
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
    unique_mapping = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes**2)
    confmat = bins.reshape(num_classes, num_classes)
    return confmat


def _confusion_matrix_compute(confmat: torch.Tensor, normalize: Optional[str] = None) -> torch.Tensor:
    allowed_normalize = ('true', 'pred', 'all', 'none', None)
    assert normalize in allowed_normalize, \
        f"Argument average needs to one of the following: {allowed_normalize}"
    confmat = confmat.float()
    if normalize is not None and normalize != 'none':
        if normalize == 'true':
            cm = confmat / confmat.sum(axis=1, keepdim=True)
        elif normalize == 'pred':
            cm = confmat / confmat.sum(axis=0, keepdim=True)
        elif normalize == 'all':
            cm = confmat / confmat.sum()
        nan_elements = cm[torch.isnan(cm)].nelement()
        if nan_elements != 0:
            cm[torch.isnan(cm)] = 0
            rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
        return cm
    return confmat


def confusion_matrix(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    normalize: Optional[str] = None,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Computes the confusion matrix. Works with binary, multiclass, and multilabel data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        normalize: Normalization mode for confusion matrix. Choose from

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5

    Example:

        >>> from pytorch_lightning.metrics.functional import confusion_matrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confusion_matrix(preds, target, num_classes=2)
        tensor([[2., 0.],
                [1., 1.]])
    """
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold)
    return _confusion_matrix_compute(confmat, normalize)
