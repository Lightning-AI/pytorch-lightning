from functools import wraps
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from pytorch_lightning.metrics.functional.reduction import reduce
from pytorch_lightning.utilities import FLOAT16_EPSILON, rank_zero_warn


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
        rank_zero_warn(f'You have set {num_classes} number of classes if different from'
                       f' predicted ({num_pred_classes}) and target ({num_target_classes}) number of classes')
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
    Calculates the number of true postive, false postive, true negative
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

    return tps, fps, tns, fns, sups


def accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction='elementwise_mean',
) -> torch.Tensor:
    """
    Computes the accuracy classification score

    Args:
        pred: predicted labels
        target: ground truth labels
        num_classes: number of classes
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
         A Tensor with the classification score.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> accuracy(x, y)
        tensor(0.7500)

    """
    if not (target > 0).any() and num_classes is None:
        raise RuntimeError("cannot infer num_classes when target is all zero")

    tps, fps, tns, fns, sups = stat_scores_multiple_classes(
        pred=pred, target=target, num_classes=num_classes, reduction=reduction)

    return tps / sups


def confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = False,
) -> torch.Tensor:
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Args:
        pred: estimated targets
        target: ground truth labels
        normalize: normalizes confusion matrix

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
    num_classes = get_num_classes(pred, target, None)

    unique_labels = target.view(-1) * num_classes + pred.view(-1)

    bins = torch.bincount(unique_labels, minlength=num_classes ** 2)
    cm = bins.reshape(num_classes, num_classes).squeeze().float()

    if normalize:
        cm = cm / cm.sum(-1)

    return cm


def precision_recall(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction: str = 'elementwise_mean',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes precision and recall for different thresholds

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with precision and recall

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> precision_recall(x, y)
        (tensor(0.7500), tensor(0.6250))

    """
    tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred=pred, target=target, num_classes=num_classes)

    tps = tps.to(torch.float)
    fps = fps.to(torch.float)
    fns = fns.to(torch.float)

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)

    # solution by justus, see https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/9
    precision[precision != precision] = 0
    recall[recall != recall] = 0

    precision = reduce(precision, reduction=reduction)
    recall = reduce(recall, reduction=reduction)
    return precision, recall


def precision(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    Computes precision score.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with precision.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> precision(x, y)
        tensor(0.7500)

    """
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, reduction=reduction)[0]


def recall(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    """
    Computes recall score.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        Tensor with recall.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> recall(x, y)
        tensor(0.6250)
    """
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, reduction=reduction)[1]


def fbeta_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        beta: float,
        num_classes: Optional[int] = None,
        reduction: str = 'elementwise_mean',
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
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements.

    Return:
        Tensor with the value of F-score. It is a value between 0-1.

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> fbeta_score(x, y, 0.2)
        tensor(0.7407)
    """
    prec, rec = precision_recall(pred=pred, target=target,
                                 num_classes=num_classes,
                                 reduction='none')

    nom = (1 + beta ** 2) * prec * rec
    denom = ((beta ** 2) * prec + rec)
    fbeta = nom / denom

    # drop NaN after zero division
    fbeta[fbeta != fbeta] = 0

    return reduce(fbeta, reduction=reduction)


def f1_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction='elementwise_mean',
) -> torch.Tensor:
    """
    Computes the F1-score (a.k.a F-measure), which is the harmonic mean of the precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements.

    Return:
         Tensor containing F1-score

    Example:

        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> f1_score(x, y)
        tensor(0.6667)
    """
    return fbeta_score(pred=pred, target=target, beta=1.,
                       num_classes=num_classes, reduction=reduction)


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
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> fpr, tpr, thresholds = roc(x, y)
        >>> fpr
        tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000])
        >>> tpr
        tensor([0., 0., 0., 1., 1.])
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
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target)
        >>> precision
        tensor([0.3333, 0.0000, 0.0000, 1.0000])
        >>> recall
        tensor([1., 0., 0., 0.])
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
        reorder: reorder coordinates, so they are increasing

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
        # can't use lexsort here since it is not implemented for torch
        order = torch.argsort(x)
        x, y = x[order], y[order]
    else:
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx, 0).all():
                direction = -1.
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

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
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auroc(x, y)
        tensor(0.3333)
    """

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
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

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
        num_classes: Optional[int] = None,
        remove_bg: bool = False,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Intersection over union, or Jaccard index calculation.

    Args:
        pred: Tensor containing predictions
        target: Tensor containing targets
        num_classes: Optionally specify the number of classes
        remove_bg: Flag to state whether a background class has been included
            within input parameters. If true, will remove background class. If
            false, return IoU over all classes
            Assumes that background is '0' class in input tensor
        reduction: a method to reduce metric score over labels (default: takes the mean)
            Available reduction methods:

            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Return:
        IoU score : Tensor containing single value if reduction is
        'elementwise_mean', or number of classes if reduction is 'none'

    Example:

        >>> target = torch.randint(0, 1, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou(pred, target)
        tensor(0.4914)

    """
    tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, target, num_classes)
    if remove_bg:
        tps = tps[1:]
        fps = fps[1:]
        fns = fns[1:]
    denom = fps + fns + tps
    denom[denom == 0] = torch.tensor(FLOAT16_EPSILON).type_as(denom)
    iou = tps / denom
    return reduce(iou, reduction=reduction)
