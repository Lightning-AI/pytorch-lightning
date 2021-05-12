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
from functools import wraps
from typing import Callable, Optional, Sequence, Tuple

import torch
from torchmetrics.utilities import class_reduce, reduce
from torchmetrics.utilities.data import get_num_classes, to_categorical

from pytorch_lightning.metrics.functional.auc import auc as __auc
from pytorch_lightning.metrics.functional.auroc import auroc as __auroc
from pytorch_lightning.metrics.functional.iou import iou as __iou
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn


def stat_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.stat_scores`. Will be removed in v1.4.0.
    """
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((pred == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((pred != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((pred != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def _confmat_normalize(cm):
    """ Normalization function for confusion matrix """
    cm = cm / cm.sum(-1, keepdim=True)
    nan_elements = cm[torch.isnan(cm)].nelement()
    if nan_elements != 0:
        cm[torch.isnan(cm)] = 0
        rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
    return cm


# todo: remove in 1.4
def auc(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.auc`. Will be removed in v1.4.0.
    """
    rank_zero_deprecation(
        "This `auc` was deprecated in v1.2.0 in favor of"
        " `pytorch_lightning.metrics.functional.auc import auc`."
        " It will be removed in v1.4.0"
    )
    return __auc(x, y)


# todo: remove in 1.4
def _auc_decorator() -> Callable:

    def wrapper(func_to_decorate: Callable) -> Callable:

        @wraps(func_to_decorate)
        def new_func(*args, **kwargs) -> torch.Tensor:
            x, y = func_to_decorate(*args, **kwargs)[:2]

            return auc(x, y)

        return new_func

    return wrapper


# todo: remove in 1.4
def _multiclass_auc_decorator() -> Callable:

    def wrapper(func_to_decorate: Callable) -> Callable:

        @wraps(func_to_decorate)
        def new_func(*args, **kwargs) -> torch.Tensor:
            results = []
            for class_result in func_to_decorate(*args, **kwargs):
                x, y = class_result[:2]
                results.append(auc(x, y))

            return torch.stack(results)

        return new_func

    return wrapper


# todo: remove in 1.4
def auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
    max_fpr: float = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.auroc`. Will be removed in v1.4.0.
    """
    rank_zero_deprecation(
        "This `auroc` was deprecated in v1.2.0 in favor of `pytorch_lightning.metrics.functional.auroc import auroc`."
        " It will be removed in v1.4.0"
    )
    return __auroc(
        preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label, max_fpr=max_fpr, num_classes=1
    )


# todo: remove in 1.4
def multiclass_auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.auroc`. Will be removed in v1.4.0.
    """
    rank_zero_deprecation(
        "This `multiclass_auroc` was deprecated in v1.2.0 in favor of"
        " `pytorch_lightning.metrics.functional.auroc import auroc`."
        " It will be removed in v1.4.0"
    )

    return __auroc(preds=pred, target=target, sample_weights=sample_weight, num_classes=num_classes)


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.dice_score`. Will be removed in v1.4.0.
    """
    num_classes = pred.shape[1]
    bg = (1 - int(bool(bg)))
    scores = torch.zeros(num_classes - bg, device=pred.device, dtype=torch.float32)
    for i in range(bg, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg] += no_fg_score
            continue

        tp, fp, tn, fn, sup = stat_scores(pred=pred, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score

        scores[i - bg] += score_cls
    return reduce(scores, reduction=reduction)


# todo: remove in 1.4
def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    num_classes: Optional[int] = None,
    reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.iou`. Will be removed in v1.4.0.
    """
    rank_zero_deprecation(
        "This `iou` was deprecated in v1.2.0 in favor of `from pytorch_lightning.metrics.functional.iou import iou`."
        " It will be removed in v1.4.0"
    )
    return __iou(
        pred,
        target,
        ignore_index=ignore_index,
        absent_score=absent_score,
        threshold=0.5,
        num_classes=num_classes,
        reduction=reduction
    )
