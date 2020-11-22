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
from torch.nn import functional as F
from pytorch_lightning.utilities import rank_zero_warn


def to_onehot(
    tensor: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Converts a dense label tensor to one-hot format

    Args:
        tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    """
    if num_classes is None:
        num_classes = int(tensor.max().detach().item() + 1)
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], num_classes, *shape[1:], dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    Converts a tensor of probabilities to a dense label tensor

    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:

        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])

    """
    return torch.argmax(tensor, dim=argmax_dim)


def get_num_classes(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
) -> int:
    """
    Calculates the number of classes for a given prediction and target tensor.

    Args:
        pred: predicted values
        target: true labels
        num_classes: number of classes if known

    Return:
        An integer that represents the number of classes.
    """
    num_target_classes = int(target.max().detach().item() + 1)
    num_pred_classes = int(pred.max().detach().item() + 1)
    num_all_classes = max(num_target_classes, num_pred_classes)

    if num_classes is None:
        num_classes = num_all_classes
    elif num_classes != num_all_classes:
        rank_zero_warn(
            f"You have set {num_classes} number of classes which is"
            f" different from predicted ({num_pred_classes}) and"
            f" target ({num_target_classes}) number of classes",
            RuntimeWarning,
        )
    return num_classes


def _confmat_normalize(cm):
    """ Normalization function for confusion matrix """
    cm = cm / cm.sum(-1, keepdim=True)
    nan_elements = cm[torch.isnan(cm)].nelement()
    if nan_elements != 0:
        cm[torch.isnan(cm)] = 0
        rank_zero_warn(f"{nan_elements} nan values found in confusion matrix have been replaced with zeros.")
    return cm


def _binary_clf_curve(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
    """
    if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
        sample_weight = torch.tensor(sample_weight, device=pred.device, dtype=torch.float)

    # remove class dimension if necessary
    if pred.ndim > target.ndim:
        pred = pred[:, 0]
    desc_score_indices = torch.argsort(pred, descending=True)

    pred = pred[desc_score_indices]
    target = target[desc_score_indices]

    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(pred[1:] - pred[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, (0, 1), value=target.size(0) - 1)

    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, pred[threshold_idxs]


def roc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Receiver Operating Characteristic (ROC). It assumes classifier is binary.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class

    Return:
        false-positive rate (fpr), true-positive rate (tpr), thresholds

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 1, 1])
        >>> fpr, tpr, thresholds = roc(x, y)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([4, 3, 2, 1, 0])

    """
    fps, tps, thresholds = _binary_clf_curve(pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label)

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


def multiclass_roc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Computes the Receiver Operating Characteristic (ROC) for multiclass predictors.

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
    num_classes = get_num_classes(pred, target, num_classes)

    class_roc_vals = []
    for c in range(num_classes):
        pred_c = pred[:, c]

        class_roc_vals.append(roc(pred=pred_c, target=target, sample_weight=sample_weight, pos_label=c))

    return tuple(class_roc_vals)


def precision_recall_curve(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes precision-recall pairs for different thresholds.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class

    Return:
         precision, recall, thresholds

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target)
        >>> precision
        tensor([0.6667, 0.5000, 0.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.0000, 0.0000])
        >>> thresholds
        tensor([1, 2, 3])

    """
    fps, tps, thresholds = _binary_clf_curve(pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    sl = slice(0, last_ind.item() + 1)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat([reversed(precision[sl]), torch.ones(1, dtype=precision.dtype, device=precision.device)])

    recall = torch.cat([reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])

    thresholds = torch.tensor(reversed(thresholds[sl]))

    return precision, recall, thresholds


def multiclass_precision_recall_curve(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes precision-recall pairs for different thresholds given a multiclass scores.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weight
        num_classes: number of classes

    Return:
        number of classes, precision, recall, thresholds

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> nb_classes, precision, recall, thresholds = multiclass_precision_recall_curve(pred, target)
        >>> nb_classes
        (tensor([1., 1.]), tensor([1., 0.]), tensor([0.8500]))
        >>> precision
        (tensor([1., 1.]), tensor([1., 0.]), tensor([0.8500]))
        >>> recall
        (tensor([0.2500, 0.0000, 1.0000]), tensor([1., 0., 0.]), tensor([0.0500, 0.8500]))
        >>> thresholds   # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.2500, 0.0000, 1.0000]), tensor([1., 0., 0.]), tensor([0.0500, 0.8500]))
    """
    num_classes = get_num_classes(pred, target, num_classes)

    class_pr_vals = []
    for c in range(num_classes):
        pred_c = pred[:, c]

        class_pr_vals.append(
            precision_recall_curve(pred=pred_c, target=target, sample_weight=sample_weight, pos_label=c)
        )

    return tuple(class_pr_vals)


def auc(x: torch.Tensor, y: torch.Tensor, reorder: bool = True) -> torch.Tensor:
    """
    Computes Area Under the Curve (AUC) using the trapezoidal rule

    Args:
        x: x-coordinates
        y: y-coordinates
        reorder: reorder coordinates, so they are increasing. The unstable algorithm of torch.argsort is
            used internally to sort `x` which may in some cases cause inaccuracies in the result.
            WARNING: Deprecated and will be removed in v1.1.

    Return:
        Tensor containing AUC score (float)

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auc(x, y)
        tensor(4.)
    """
    direction = 1.0

    if reorder:
        rank_zero_warn(
            "The `reorder` parameter to `auc` has been deprecated and will be removed in v1.1"
            " Note that when `reorder` is True, the unstable algorithm of torch.argsort is"
            " used internally to sort 'x' which may in some cases cause inaccuracies"
            " in the result.",
            DeprecationWarning,
        )
        # can't use lexsort here since it is not implemented for torch
        order = torch.argsort(x)
        x, y = x[order], y[order]
    else:
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx, 0).all():
                direction = -1.0
            else:
                # TODO: Update message on removing reorder
                raise ValueError(
                    "Reorder is not turned on, and the 'x' array is" f" neither increasing or decreasing: {x}"
                )

    return direction * torch.trapz(y, x)


def auc_decorator(reorder: bool = True) -> Callable:
    def wrapper(func_to_decorate: Callable) -> Callable:
        @wraps(func_to_decorate)
        def new_func(*args, **kwargs) -> torch.Tensor:
            x, y = func_to_decorate(*args, **kwargs)[:2]

            return auc(x, y, reorder=reorder)

        return new_func

    return wrapper


def multiclass_auc_decorator(reorder: bool = True) -> Callable:
    def wrapper(func_to_decorate: Callable) -> Callable:
        @wraps(func_to_decorate)
        def new_func(*args, **kwargs) -> torch.Tensor:
            results = []
            for class_result in func_to_decorate(*args, **kwargs):
                x, y = class_result[:2]
                results.append(auc(x, y, reorder=reorder))

            return torch.stack(results)

        return new_func

    return wrapper


def auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.0,
) -> torch.Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class

    Return:
        Tensor containing ROCAUC score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 1, 0])
        >>> auroc(x, y)
        tensor(0.5000)
    """
    if any(target > 1):
        raise ValueError(
            "AUROC metric is meant for binary classification, but"
            " target tensor contains value different from 0 and 1."
            " Use `multiclass_auroc` for multi class classification."
        )

    @auc_decorator(reorder=True)
    def _auroc(pred, target, sample_weight, pos_label):
        return roc(pred, target, sample_weight, pos_label)

    return _auroc(pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label)


def multiclass_auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from multiclass
    prediction scores

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
        >>> multiclass_auroc(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        tensor(0.6667)
    """
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

    @multiclass_auc_decorator(reorder=False)
    def _multiclass_auroc(pred, target, sample_weight, num_classes):
        return multiclass_roc(pred, target, sample_weight, num_classes)

    class_aurocs = _multiclass_auroc(pred=pred, target=target, sample_weight=sample_weight, num_classes=num_classes)
    return torch.mean(class_aurocs)


def average_precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    pos_label: int = 1.0,
) -> torch.Tensor:
    """
    Compute average precision from prediction scores

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class

    Return:
        Tensor containing average precision score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> average_precision(x, y)
        tensor(0.3333)
    """
    precision, recall, _ = precision_recall_curve(
        pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label
    )
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
