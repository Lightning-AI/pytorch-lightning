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
from pytorch_lightning.metrics.functional.reduction import class_reduce, reduce
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
    tensor_onehot = torch.zeros(shape[0], num_classes, *shape[1:],
                                dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def to_categorical(
        tensor: torch.Tensor,
        argmax_dim: int = 1
) -> torch.Tensor:
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
        rank_zero_warn(f'You have set {num_classes} number of classes which is'
                       f' different from predicted ({num_pred_classes}) and'
                       f' target ({num_target_classes}) number of classes',
                       RuntimeWarning)
    return num_classes


def stat_scores(
        pred: torch.Tensor,
        target: torch.Tensor,
        class_index: int, argmax_dim: int = 1,
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

    Args:
        pred: prediction tensor
        target: target tensor
        num_classes: number of classes if known
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over
        reduction: a method to reduce metric score over labels (default: none)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tps, fps, tns, fns, sups = stat_scores_multiple_classes(x, y)
        >>> tps
        tensor([0., 0., 1., 1.])
        >>> fps
        tensor([0., 1., 0., 0.])
        >>> tns
        tensor([2., 2., 2., 2.])
        >>> fns
        tensor([1., 0., 0., 0.])
        >>> sups
        tensor([1., 0., 1., 1.])

    """
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

        tps = torch.zeros((num_classes + 1,), device=pred.device)
        fps = torch.zeros((num_classes + 1,), device=pred.device)
        tns = torch.zeros((num_classes + 1,), device=pred.device)
        fns = torch.zeros((num_classes + 1,), device=pred.device)
        sups = torch.zeros((num_classes + 1,), device=pred.device)

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


def accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
        return_state: bool = False
) -> torch.Tensor:
    """
    Computes the accuracy classification score

    Args:
        pred: predicted labels
        target: ground truth labels
        num_classes: number of classes
        class_reduction: method to reduce metric score over labels

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class
        return_state: returns a internal state that can be ddp reduced
            before doing the final calculation

    Return:
         A Tensor with the accuracy score.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> accuracy(x, y)
        tensor(0.7500)

    """
    tps, fps, tns, fns, sups = stat_scores_multiple_classes(
        pred=pred, target=target, num_classes=num_classes)
    if return_state:
        return {'tps': tps, 'sups': sups}
    return class_reduce(tps, sups, sups, class_reduction=class_reduction)


def _confmat_normalize(cm):
    """ Normalization function for confusion matrix """
    cm = cm / cm.sum(-1, keepdim=True)
    nan_elements = cm[torch.isnan(cm)].nelement()
    if nan_elements != 0:
        cm[torch.isnan(cm)] = 0
        rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
    return cm


def confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = False,
        num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Args:
        pred: estimated targets
        target: ground truth labels
        normalize: normalizes confusion matrix
        num_classes: number of classes

    Return:
        Tensor, confusion matrix C [num_classes, num_classes ]

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> confusion_matrix(x, y)
        tensor([[0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])
    """
    num_classes = get_num_classes(pred, target, num_classes)

    unique_labels = (target.view(-1) * num_classes + pred.view(-1)).to(torch.int)

    bins = torch.bincount(unique_labels, minlength=num_classes ** 2)
    cm = bins.reshape(num_classes, num_classes).squeeze().float()

    if normalize:
        cm = _confmat_normalize(cm)

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
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, class_reduction=class_reduction)[0]


def recall(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
) -> torch.Tensor:
    """
    Computes recall score.

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
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, class_reduction=class_reduction)[1]


def fbeta_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        beta: float,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
) -> torch.Tensor:
    """
    Computes the F-beta score which is a weighted harmonic mean of precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        beta: weights recall when combining the score.
            beta < 1: more weight to precision.
            beta > 1 more weight to recall
            beta = 0: only precision
            beta -> inf: only recall
        num_classes: number of classes
        class_reduction: method to reduce metric score over labels

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'``: returns calculated metric per class

    Return:
        Tensor with the value of F-score. It is a value between 0-1.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> fbeta_score(x, y, 0.2)
        tensor(0.7500)
    """
    # We need to differentiate at which point to do class reduction
    intermidiate_reduction = 'none' if class_reduction != "micro" else 'micro'

    prec, rec, sups = precision_recall(pred=pred, target=target,
                                       num_classes=num_classes,
                                       class_reduction=intermidiate_reduction,
                                       return_support=True)
    num = (1 + beta ** 2) * prec * rec
    denom = ((beta ** 2) * prec + rec)
    if intermidiate_reduction == 'micro':
        return torch.sum(num) / torch.sum(denom)
    return class_reduce(num, denom, sups, class_reduction=class_reduction)


def f1_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
) -> torch.Tensor:
    """
    Computes the F1-score (a.k.a F-measure), which is the harmonic mean of the precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

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
         Tensor containing F1-score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> f1_score(x, y)
        tensor(0.7500)
    """
    return fbeta_score(pred=pred, target=target, beta=1.,
                       num_classes=num_classes, class_reduction=class_reduction)


def _binary_clf_curve(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
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
        weight = 1.

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
        pos_label: int = 1.,
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
    fps, tps, thresholds = _binary_clf_curve(pred=pred, target=target,
                                             sample_weight=sample_weight,
                                             pos_label=pos_label)

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

        class_roc_vals.append(roc(pred=pred_c, target=target,
                                  sample_weight=sample_weight, pos_label=c))

    return tuple(class_roc_vals)


def precision_recall_curve(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
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
    fps, tps, thresholds = _binary_clf_curve(pred=pred, target=target,
                                             sample_weight=sample_weight,
                                             pos_label=pos_label)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    sl = slice(0, last_ind.item() + 1)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat([reversed(precision[sl]),
                           torch.ones(1, dtype=precision.dtype,
                                      device=precision.device)])

    recall = torch.cat([reversed(recall[sl]),
                        torch.zeros(1, dtype=recall.dtype,
                                    device=recall.device)])

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

        class_pr_vals.append(precision_recall_curve(
            pred=pred_c,
            target=target,
            sample_weight=sample_weight, pos_label=c))

    return tuple(class_pr_vals)


def auc(
        x: torch.Tensor,
        y: torch.Tensor,
        reorder: bool = True
) -> torch.Tensor:
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
    direction = 1.

    if reorder:
        rank_zero_warn("The `reorder` parameter to `auc` has been deprecated and will be removed in v1.1"
                       " Note that when `reorder` is True, the unstable algorithm of torch.argsort is"
                       " used internally to sort 'x' which may in some cases cause inaccuracies"
                       " in the result.",
                       DeprecationWarning)
        # can't use lexsort here since it is not implemented for torch
        order = torch.argsort(x)
        x, y = x[order], y[order]
    else:
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx, 0).all():
                direction = -1.
            else:
                # TODO: Update message on removing reorder
                raise ValueError("Reorder is not turned on, and the 'x' array is"
                                 f" neither increasing or decreasing: {x}")

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
        def new_func(*args, **kwargs) -> torch.Tensor:
            results = []
            for class_result in func_to_decorate(*args, **kwargs):
                x, y = class_result[:2]
                results.append(auc(x, y, reorder=reorder))

            return torch.cat(results)

        return new_func

    return wrapper


def auroc(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
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
        raise ValueError('AUROC metric is meant for binary classification, but'
                         ' target tensor contains value different from 0 and 1.'
                         ' Multiclass is currently not supported.')

    @auc_decorator(reorder=True)
    def _auroc(pred, target, sample_weight, pos_label):
        return roc(pred, target, sample_weight, pos_label)

    return _auroc(pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label)


def average_precision(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
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
    precision, recall, _ = precision_recall_curve(pred=pred, target=target,
                                                  sample_weight=sample_weight,
                                                  pos_label=pos_label)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])


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
    if pred.size() != target.size():
        raise ValueError(f"'pred' shape ({pred.size()}) must equal 'target' shape ({target.size()})")

    if not torch.allclose(pred.float(), pred.int().float()):
        raise ValueError("'pred' must contain integer targets.")

    num_classes = get_num_classes(pred=pred, target=target, num_classes=num_classes)

    tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, target, num_classes)

    scores = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)

    for class_idx in range(num_classes):
        if class_idx == ignore_index:
            continue

        tp = tps[class_idx]
        fp = fps[class_idx]
        fn = fns[class_idx]
        sup = sups[class_idx]

        # If this class is absent in the target (no support) AND absent in the pred (no true or false
        # positives), then use the absent_score for this class.
        if sup + tp + fp == 0:
            scores[class_idx] = absent_score
            continue

        denom = tp + fp + fn
        # Note that we do not need to worry about division-by-zero here since we know (sup + tp + fp != 0) from above,
        # which means ((tp+fn) + tp + fp != 0), which means (2tp + fp + fn != 0). Since all vars are non-negative, we
        # can conclude (tp + fp + fn > 0), meaning the denominator is non-zero for each class.
        score = tp.to(torch.float) / denom
        scores[class_idx] = score

    # Remove the ignored class index from the scores.
    if ignore_index is not None and ignore_index >= 0 and ignore_index < num_classes:
        scores = torch.cat([
            scores[:ignore_index],
            scores[ignore_index + 1:],
        ])

    return reduce(scores, reduction=reduction)
