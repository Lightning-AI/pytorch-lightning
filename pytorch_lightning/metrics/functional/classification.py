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

from pytorch_lightning.metrics.functional.auc import auc as __auc
from pytorch_lightning.metrics.functional.auroc import auroc as __auroc
from pytorch_lightning.metrics.functional.average_precision import average_precision as __ap
from pytorch_lightning.metrics.functional.iou import iou as __iou
from pytorch_lightning.metrics.functional.precision_recall_curve import _binary_clf_curve
from pytorch_lightning.metrics.functional.precision_recall_curve import precision_recall_curve as __prc
from pytorch_lightning.metrics.functional.roc import roc as __roc
from pytorch_lightning.metrics.utils import class_reduce
from pytorch_lightning.metrics.utils import get_num_classes as __gnc
from pytorch_lightning.metrics.utils import reduce
from pytorch_lightning.metrics.utils import to_categorical as __tc
from pytorch_lightning.metrics.utils import to_onehot as __to
from pytorch_lightning.utilities import rank_zero_warn


def to_onehot(
    tensor: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Converts a dense label tensor to one-hot format

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.utils.to_onehot`
    """
    rank_zero_warn(
        "This `to_onehot` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.utils import to_onehot`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __to(tensor, num_classes)


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    Converts a tensor of probabilities to a dense label tensor

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.utils.to_categorical`

    """
    rank_zero_warn(
        "This `to_categorical` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.utils import to_categorical`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __tc(tensor)


def get_num_classes(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
) -> int:
    """
    Calculates the number of classes for a given prediction and target tensor.

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.utils.get_num_classes`

    """
    rank_zero_warn(
        "This `get_num_classes` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.utils import get_num_classes`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __gnc(pred, target, num_classes)


def stat_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the number of true positive, false positive, true negative
    and false negative for a specific class

    Args:
        pred: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))

    """
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((pred == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((pred != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((pred != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def stat_scores_multiple_classes(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    argmax_dim: int = 1,
    reduction: str = 'none',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the number of true positive, false positive, true negative
    and false negative for each class

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.functional.stat_scores`

    """

    rank_zero_warn(
        "This `stat_scores_multiple_classes` was deprecated in v1.2.0 in favor of"
        " `from pytorch_lightning.metrics.functional import stat_scores`."
        " It will be removed in v1.4.0", DeprecationWarning
    )
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    num_classes = get_num_classes(pred=pred, target=target, num_classes=num_classes)

    if pred.dtype != torch.bool:
        pred = pred.clamp_max(max=num_classes)
    if target.dtype != torch.bool:
        target = target.clamp_max(max=num_classes)

    possible_reductions = ('none', 'sum', 'elementwise_mean')
    if reduction not in possible_reductions:
        raise ValueError("reduction type %s not supported" % reduction)

    if reduction == 'none':
        pred = pred.view((-1, )).long()
        target = target.view((-1, )).long()

        tps = torch.zeros((num_classes + 1, ), device=pred.device)
        fps = torch.zeros((num_classes + 1, ), device=pred.device)
        fns = torch.zeros((num_classes + 1, ), device=pred.device)
        sups = torch.zeros((num_classes + 1, ), device=pred.device)

        match_true = (pred == target).float()
        match_false = 1 - match_true

        tps.scatter_add_(0, pred, match_true)
        fps.scatter_add_(0, pred, match_false)
        fns.scatter_add_(0, target, match_false)
        tns = pred.size(0) - (tps + fps + fns)
        sups.scatter_add_(0, target, torch.ones_like(match_true))

        tps = tps[:num_classes]
        fps = fps[:num_classes]
        tns = tns[:num_classes]
        fns = fns[:num_classes]
        sups = sups[:num_classes]

    elif reduction == 'sum' or reduction == 'elementwise_mean':
        count_match_true = (pred == target).sum().float()
        oob_tp, oob_fp, oob_tn, oob_fn, oob_sup = stat_scores(pred, target, num_classes, argmax_dim)

        tps = count_match_true - oob_tp
        fps = pred.nelement() - count_match_true - oob_fp
        fns = pred.nelement() - count_match_true - oob_fn
        tns = pred.nelement() * (num_classes + 1) - (tps + fps + fns + oob_tn)
        sups = pred.nelement() - oob_sup.float()

        if reduction == 'elementwise_mean':
            tps /= num_classes
            fps /= num_classes
            fns /= num_classes
            tns /= num_classes
            sups /= num_classes

    return tps.float(), fps.float(), tns.float(), fns.float(), sups.float()


def _confmat_normalize(cm):
    """ Normalization function for confusion matrix """
    cm = cm / cm.sum(-1, keepdim=True)
    nan_elements = cm[torch.isnan(cm)].nelement()
    if nan_elements != 0:
        cm[torch.isnan(cm)] = 0
        rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
    return cm


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    class_reduction: str = 'micro',
    return_support: bool = False,
    return_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes precision and recall for different thresholds

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.precision_recall`.
     Will be removed in v1.4.0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        class_reduction: method to reduce metric score over labels

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class

        return_support: returns the support for each class, need for fbeta/f1 calculations
        return_state: returns a internal state that can be ddp reduced
            before doing the final calculation

    Return:
        Tensor with precision and recall

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 2, 2, 2])
        >>> precision_recall(x, y, class_reduction='macro')
        (tensor(0.5000), tensor(0.3333))

    """
    rank_zero_warn(
        "This `precision_recall` was deprecated in v1.2.0 in favor of"
        " `from pytorch_lightning.metrcs.functional import precision_recall`."
        " It will be removed in v1.4.0", DeprecationWarning
    )

    tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred=pred, target=target, num_classes=num_classes)

    precision = class_reduce(tps, tps + fps, sups, class_reduction=class_reduction)
    recall = class_reduce(tps, tps + fns, sups, class_reduction=class_reduction)
    if return_state:
        return {'tps': tps, 'fps': fps, 'fns': fns, 'sups': sups}
    if return_support:
        return precision, recall, sups
    return precision, recall


def precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    class_reduction: str = 'micro',
) -> torch.Tensor:
    """
    Computes precision score.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.recall`. Will be removed in v1.4.0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        class_reduction: method to reduce metric score over labels

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class

    Return:
        Tensor with precision.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> precision(x, y)
        tensor(0.7500)

    """
    rank_zero_warn(
        "This `precision` was deprecated in v1.2.0 in favor of"
        " `from pytorch_lightning.metrics.functional import precision`."
        " It will be removed in v1.4.0", DeprecationWarning
    )

    return precision_recall(pred=pred, target=target, num_classes=num_classes, class_reduction=class_reduction)[0]


def recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    class_reduction: str = 'micro',
) -> torch.Tensor:
    """
    Computes recall score.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.recall`. Will be removed in v1.4.0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        class_reduction: method to reduce metric score over labels

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class

    Return:
        Tensor with recall.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> recall(x, y)
        tensor(0.7500)
    """
    rank_zero_warn(
        "This `recall` was deprecated in v1.2.0 in favor of"
        " `from pytorch_lightning.metrics.functional import recall`."
        " It will be removed in v1.4.0", DeprecationWarning
    )

    return precision_recall(pred=pred, target=target, num_classes=num_classes, class_reduction=class_reduction)[1]


# todo: remove in 1.3
def roc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Receiver Operating Characteristic (ROC). It assumes classifier is binary.

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.functional.roc.roc`
    """
    rank_zero_warn(
        "This `multiclass_roc` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.functional.roc import roc`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __roc(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)


# TODO: deprecated in favor of general ROC in pytorch_lightning/metrics/functional/roc.py
def _roc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Receiver Operating Characteristic (ROC). It assumes classifier is binary.

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.functional.roc.roc`

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 1, 1])
        >>> fpr, tpr, thresholds = _roc(x, y)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([4, 3, 2, 1, 0])

    """
    rank_zero_warn(
        "This `multiclass_roc` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.functional.roc import roc`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    fps, tps, thresholds = _binary_clf_curve(pred, target, sample_weights=sample_weight, pos_label=pos_label)

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
    fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
    thresholds = torch.cat([thresholds[0][None] + 1, thresholds])

    if fps[-1] <= 0:
        raise ValueError("No negative samples in targets, false positive value should be meaningless")

    fpr = fps / fps[-1]

    if tps[-1] <= 0:
        raise ValueError("No positive samples in targets, true positive value should be meaningless")

    tpr = tps / tps[-1]

    return fpr, tpr, thresholds


# TODO: deprecated in favor of general ROC in pytorch_lightning/metrics/functional/roc.py
def multiclass_roc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Computes the Receiver Operating Characteristic (ROC) for multiclass predictors.

    .. warning :: Deprecated in favor of :func:`~pytorch_lightning.metrics.functional.roc.roc`

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        num_classes: number of classes (default: None, computes automatically from data)

    Return:
        returns roc for each class.
        Number of classes, false-positive rate (fpr), true-positive rate (tpr), thresholds

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> multiclass_roc(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        ((tensor([0., 0., 1.]), tensor([0., 1., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0., 0., 1.]), tensor([0., 1., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0.0000, 0.3333, 1.0000]), tensor([0., 0., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0.0000, 0.3333, 1.0000]), tensor([0., 0., 1.]), tensor([1.8500, 0.8500, 0.0500])))
    """
    rank_zero_warn(
        "This `multiclass_roc` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.functional.roc import roc`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    num_classes = get_num_classes(pred, target, num_classes)

    class_roc_vals = []
    for c in range(num_classes):
        pred_c = pred[:, c]

        class_roc_vals.append(_roc(pred=pred_c, target=target, sample_weight=sample_weight, pos_label=c))

    return tuple(class_roc_vals)


def auc(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Area Under the Curve (AUC) using the trapezoidal rule

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.auc.auc`. Will be removed
     in v1.4.0.

    Args:
        x: x-coordinates
        y: y-coordinates

    Return:
        Tensor containing AUC score (float)

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auc(x, y)
        tensor(4.)
    """
    rank_zero_warn(
        "This `auc` was deprecated in v1.2.0 in favor of"
        " `pytorch_lightning.metrics.functional.auc import auc`."
        " It will be removed in v1.4.0", DeprecationWarning
    )
    return __auc(x, y)


def auc_decorator() -> Callable:
    rank_zero_warn("This `auc_decorator` was deprecated in v1.2.0." " It will be removed in v1.4.0", DeprecationWarning)

    def wrapper(func_to_decorate: Callable) -> Callable:

        @wraps(func_to_decorate)
        def new_func(*args, **kwargs) -> torch.Tensor:
            x, y = func_to_decorate(*args, **kwargs)[:2]

            return auc(x, y)

        return new_func

    return wrapper


def multiclass_auc_decorator() -> Callable:
    rank_zero_warn(
        "This `multiclass_auc_decorator` was deprecated in v1.2.0."
        " It will be removed in v1.4.0", DeprecationWarning
    )

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


def auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
    max_fpr: float = None,
) -> torch.Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.auroc.auroc`. Will be removed
     in v1.4.0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class
        max_fpr: If not ``None``, calculates standardized partial AUC over the
            range [0, max_fpr]. Should be a float between 0 and 1.

    Return:
        Tensor containing ROCAUC score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 1, 0])
        >>> auroc(x, y)
        tensor(0.5000)
    """
    rank_zero_warn(
        "This `auroc` was deprecated in v1.2.0 in favor of"
        " `pytorch_lightning.metrics.functional.auroc import auroc`."
        " It will be removed in v1.4.0", DeprecationWarning
    )
    return __auroc(
        preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label, max_fpr=max_fpr, num_classes=1
    )


def multiclass_auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from multiclass
    prediction scores

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.auroc.auroc`. Will be removed
     in v1.4.0.

    Args:
        pred: estimated probabilities, with shape [N, C]
        target: ground-truth labels, with shape [N,]
        sample_weight: sample weights
        num_classes: number of classes (default: None, computes automatically from data)

    Return:
        Tensor containing ROCAUC score

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> multiclass_auroc(pred, target, num_classes=4)
        tensor(0.6667)
    """
    rank_zero_warn(
        "This `multiclass_auroc` was deprecated in v1.2.0 in favor of"
        " `pytorch_lightning.metrics.functional.auroc import auroc`."
        " It will be removed in v1.4.0", DeprecationWarning
    )

    if not torch.allclose(pred.sum(dim=1), torch.tensor(1.0)):
        raise ValueError(
            "Multiclass AUROC metric expects the target scores to be"
            " probabilities, i.e. they should sum up to 1.0 over classes"
        )

    if torch.unique(target).size(0) != pred.size(1):
        raise ValueError(
            f"Number of classes found in in 'target' ({torch.unique(target).size(0)})"
            f" does not equal the number of columns in 'pred' ({pred.size(1)})."
            " Multiclass AUROC is not defined when all of the classes do not"
            " occur in the target labels."
        )

    if num_classes is not None and num_classes != pred.size(1):
        raise ValueError(
            f"Number of classes deduced from 'pred' ({pred.size(1)}) does not equal"
            f" the number of classes passed in 'num_classes' ({num_classes})."
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
    Intersection over union, or Jaccard index calculation.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.iou.iou`. Will be removed in
     v1.4.0.

    Args:
        pred: Tensor containing integer predictions, with shape [N, d1, d2, ...]
        target: Tensor containing integer targets, with shape [N, d1, d2, ...]
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1], where num_classes is either given or derived from pred and target. By default, no
            index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of the class index were present in
            `pred` AND no instances of the class index were present in `target`. For example, if we have 3 classes,
            [0, 0] for `pred`, and [0, 2] for `target`, then class 1 would be assigned the `absent_score`. Default is
            0.0.
        num_classes: Optionally specify the number of classes
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        IoU score : Tensor containing single value if reduction is
        'elementwise_mean', or number of classes if reduction is 'none'

    Example:

        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou(pred, target)
        tensor(0.9660)

    """
    rank_zero_warn(
        "This `iou` was deprecated in v1.2.0 in favor of"
        " `from pytorch_lightning.metrics.functional.iou import iou`."
        " It will be removed in v1.4.0", DeprecationWarning
    )
    return __iou(
        pred=pred,
        target=target,
        ignore_index=ignore_index,
        absent_score=absent_score,
        threshold=0.5,
        num_classes=num_classes,
        reduction=reduction
    )


# todo: remove in 1.3
def precision_recall_curve(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
):
    """
    Computes precision-recall pairs for different thresholds.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.precision_recall_curve.precision_recall_curve`
    """
    rank_zero_warn(
        "This `precision_recall_curve` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.functional.precision_recall_curve import precision_recall_curve`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __prc(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)


# todo: remove in 1.3
def multiclass_precision_recall_curve(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
):
    """
    Computes precision-recall pairs for different thresholds given a multiclass scores.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.precision_recall_curve.precision_recall_curve`
    """
    rank_zero_warn(
        "This `multiclass_precision_recall_curve` was deprecated in v1.1.0 in favor of"
        " `from pytorch_lightning.metrics.functional.precision_recall_curve import precision_recall_curve`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    if num_classes is None:
        num_classes = get_num_classes(pred, target, num_classes)
    return __prc(preds=pred, target=target, sample_weights=sample_weight, num_classes=num_classes)


# todo: remove in 1.3
def average_precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.,
):
    """
    Compute average precision from prediction scores.

    .. warning :: Deprecated in favor of
     :func:`~pytorch_lightning.metrics.functional.average_precision.average_precision`
    """
    rank_zero_warn(
        "This `average_precision` was deprecated in v1.1.0 in favor of"
        " `pytorch_lightning.metrics.functional.average_precision import average_precision`."
        " It will be removed in v1.3.0", DeprecationWarning
    )
    return __ap(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)
