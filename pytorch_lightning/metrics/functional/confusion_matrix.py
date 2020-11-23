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

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.metrics.classification.utils import _input_format_classification


def _confusion_matrix_update(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, threshold: float, logits: bool
) -> torch.Tensor:
    preds, target, _ = _input_format_classification(
        preds, target, num_classes=num_classes, is_multiclass=True, threshold=threshold, logits=logits
    )

    if len(preds.shape) > 2:
        preds = torch.movedim(preds, 1, -1).reshape(-1, num_classes)
        target = torch.movedim(target, 1, -1).reshape(-1, num_classes)

    preds = torch.argmax(preds, dim=1)
    target = torch.argmax(target, dim=1)

    unique_mapping = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes ** 2)
    confmat = bins.reshape(num_classes, num_classes)
    return confmat


def _confusion_matrix_compute(confmat: torch.Tensor, normalize: Optional[str] = None) -> torch.Tensor:
    allowed_normalize = ("true", "pred", "all", None)
    assert normalize in allowed_normalize, f"Argument average needs to one of the following: {allowed_normalize}"
    confmat = confmat.float()
    if normalize is not None:
        if normalize == "true":
            cm = confmat / confmat.sum(axis=1, keepdim=True)
        elif normalize == "pred":
            cm = confmat / confmat.sum(axis=0, keepdim=True)
        elif normalize == "all":
            cm = confmat / confmat.sum()
        nan_elements = cm[torch.isnan(cm)].nelement()
        if nan_elements != 0:
            cm[torch.isnan(cm)] = 0
            rank_zero_warn(f"{nan_elements} nan values found in confusion matrix have been replaced with zeros.")
        return cm
    return confmat


def confusion_matrix(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    normalize: Optional[str] = None,
    threshold: float = 0.5,
    logits: bool = True,
) -> torch.Tensor:
    """
    Computes the confusion matrix.

    While this metric is mainly meant for multi-class inputs, it accepts all input types
    listed in :ref:`metrics:Input types`. If you pass binary or  multi-label inputs it
    will convert them to 2-class multi-class or multi-dimensional multi-class, respectively.
    In this case use the ``threshold`` argument to control the "binarization" of predictions.

    Args:
        preds: Predictions from model (probabilities, logits, or labels)
        target: Ground truth values
        num_classes:
            Number of classes.
        normalize: Normalization mode for confusion matrix. Choose from

        - ``None``: no normalization (default)
        - ``'true'``: normalization over the targets (most commonly used)
        - ``'pred'``: normalization over the predictions
        - ``'all'``: normalization over the whole matrix

        threshold:
            Threshold probability value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. If ``logits=True``,
            this value is transformed to logits by ``logit_t = ln(t / (1-t))``. Default: 0.5
        logits:
            If predictions are floats, whether they are probabilities or logits. Default ``True``
            (predictions are logits).

    Example:

        >>> from pytorch_lightning.metrics.functional import confusion_matrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confusion_matrix(preds, target, num_classes=2)
        tensor([[2., 0.],
                [1., 1.]])
    """
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold, logits)
    return _confusion_matrix_compute(confmat, normalize)
