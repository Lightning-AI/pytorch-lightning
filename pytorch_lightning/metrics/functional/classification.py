from collections import Sequence
from functools import wraps
from typing import Optional, Tuple, Callable

import torch

from pytorch_lightning.metrics.functional.reduction import reduce


def to_onehot(
        tensor: torch.Tensor,
        n_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Converts a dense label tensor to one-hot format

    Args:
        tensor: dense label tensor, with shape [N, d1, d2, ...]

        n_classes: number of classes C

    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]
    """
    if n_classes is None:
        n_classes = int(tensor.max().detach().item() + 1)
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], n_classes, *shape[1:],
                                dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    """
    Converts a tensor of probabilities to a dense label tensor

    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply (default: 1)

    Return:
        A tensor with categorical labels [N, d2, ...]
    """
    return torch.argmax(tensor, dim=argmax_dim)


def get_num_classes(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int],
) -> int:
    """
    Returns the number of classes for a given prediction and target tensor.

        Args:
            pred: predicted values
            target: true labels
            num_classes: number of classes if known (default: None)

        Return:
            An integer that represents the number of classes.
    """
    if num_classes is None:
        if pred.ndim > target.ndim:
            num_classes = pred.size(1)
        else:
            num_classes = int(target.max().detach().item() + 1)
    return num_classes


def stat_scores(
        pred: torch.Tensor,
        target: torch.Tensor,
        class_index: int, argmax_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the number of true positive, falsepositivee, true negative
    and false negative for a specific class

    Args:
        pred: prediction tensor

        target: target tensor

        class_index: class to calculate over

        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        Tensors in the following order: True Positive, False Positive, True Negative, False Negative

    """
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((pred == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((pred != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((pred != class_index) * (target == class_index)).to(torch.long).sum()

    return tp, fp, tn, fn


def stat_scores_multiple_classes(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        argmax_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calls the stat_scores function iteratively for all classes, thus
    calculating the number of true postive, false postive, true negative
    and false negative for each class

    Args:
        pred: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        Returns tensors for: tp, fp, tn, fn

    """
    num_classes = get_num_classes(pred=pred, target=target,
                                  num_classes=num_classes)

    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tps = torch.zeros((num_classes,), device=pred.device)
    fps = torch.zeros((num_classes,), device=pred.device)
    tns = torch.zeros((num_classes,), device=pred.device)
    fns = torch.zeros((num_classes,), device=pred.device)

    for c in range(num_classes):
        tps[c], fps[c], tns[c], fns[c] = stat_scores(pred=pred, target=target,
                                                     class_index=c)

    return tps, fps, tns, fns


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
        reduction: a method for reducing accuracies over labels (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements

    Return:
         A Tensor with the classification score.
    """
    tps, fps, tns, fns = stat_scores_multiple_classes(pred=pred, target=target,
                                                      num_classes=num_classes)

    if not (target > 0).any() and num_classes is None:
        raise RuntimeError("cannot infer num_classes when target is all zero")

    accuracies = (tps + tns) / (tps + tns + fps + fns)

    return reduce(accuracies, reduction=reduction)


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
    """
    num_classes = get_num_classes(pred, target, None)

    d = target.size(-1)
    batch_vec = torch.arange(target.size(-1))
    # this will account for multilabel
    unique_labels = batch_vec * num_classes ** 2 + target.view(-1) * num_classes + pred.view(-1)

    bins = torch.bincount(unique_labels, minlength=d * num_classes ** 2)
    cm = bins.reshape(d, num_classes, num_classes).squeeze().float()

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
        reduction: method for reducing precision-recall values (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements

    Return:
        Tensor with precision and recall
    """
    tps, fps, tns, fns = stat_scores_multiple_classes(pred=pred,
                                                      target=target,
                                                      num_classes=num_classes)

    tps = tps.to(torch.float)
    fps = fps.to(torch.float)
    fns = fns.to(torch.float)

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)

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
        reduction: method for reducing precision values (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements

    Return:
        Tensor with precision.
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
        reduction: method for reducing recall values (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements

    Return:
        Tensor with recall.
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
        reduction: method for reducing F-score (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements.

    Return:
        Tensor with the value of F-score. It is a value between 0-1.
    """
    prec, rec = precision_recall(pred=pred, target=target,
                                 num_classes=num_classes,
                                 reduction='none')

    nom = (1 + beta ** 2) * prec * rec
    denom = ((beta ** 2) * prec + rec)
    fbeta = nom / denom

    return reduce(fbeta, reduction=reduction)


def f1_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        reduction='elementwise_mean',
) -> torch.Tensor:
    """
    Computes F1-score a.k.a F-measure.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        num_classes: number of classes
        reduction: method for reducing F1-score (default: takes the mean)
           Available reduction methods:

           - elementwise_mean: takes the mean
           - none: pass array
           - sum: add elements.

    Return:
         Tensor containing F1-score
    """
    return fbeta_score(pred=pred, target=target, beta=1.,
                       num_classes=num_classes, reduction=reduction)


# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
def _binary_clf_curve(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    threshold_idxs = torch.cat([distinct_value_indices,
                                torch.tensor([target.size(0) - 1])])

    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum((target == pos_label).to(torch.long) * weight,
                       dim=0)[threshold_idxs]

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
        pos_label: the label for the positive class (default: 1)

    Return:
        [Tensor, Tensor, Tensor]: false-positive rate (fpr), true-positive rate (tpr), thresholds
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
        raise ValueError("No negative samples in targets, "
                         "false positive value should be meaningless")

    fpr = fps / fps[-1]

    if tps[-1] <= 0:
        raise ValueError("No positive samples in targets, "
                         "true positive value should be meaningless")

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
        [num_classes, Tensor, Tensor, Tensor]: returns roc for each class.
        number of classes, false-positive rate (fpr), true-positive rate (tpr), thresholds
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
        pos_label: the label for the positive class (default: 1.)

    Return:
         [Tensor, Tensor, Tensor]: precision, recall, thresholds
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
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Computes precision-recall pairs for different thresholds given a multiclass scores.

    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weight
        num_classes: number of classes

    Return:
         [num_classes, Tensor, Tensor, Tensor]: number of classes, precision, recall, thresholds
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


def auc(x: torch.Tensor, y: torch.Tensor, reorder: bool = True):
    """
    Computes Area Under the Curve (AUC) using the trapezoidal rule

    Args:
        x: x-coordinates
        y: y-coordinates
        reorder: reorder coordinates, so they are increasing.

    Return:
        AUC score (float)
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


@auc_decorator(reorder=True)
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
        pos_label: the label for the positive class (default: 1.)
    """
    # fixme
    return roc(pred=pred, target=target, sample_weight=sample_weight,
               pos_label=pos_label)


def average_precision(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
) -> torch.Tensor:
    precision, recall, _ = precision_recall_curve(pred=pred, target=target,
                                                  sample_weight=sample_weight,
                                                  pos_label=pos_label)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -torch.sum(recall[1:] - recall[:-1] * precision[:-1])


def dice_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        bg: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = 'elementwise_mean',
) -> torch.Tensor:
    n_classes = pred.shape[1]
    bg = (1 - int(bool(bg)))
    scores = torch.zeros(n_classes - bg, device=pred.device, dtype=torch.float32)
    for i in range(bg, n_classes):
        if not (target == i).any():
            # no foreground class
            scores[i-bg] += no_fg_score
            continue

        tp, fp, tn, fn = stat_scores(pred=pred, target=target, class_index=i)

        denom = (2 * tp + fp + fn).to(torch.float)

        if torch.isclose(denom, torch.zeros_like(denom)).any():
            # nan result
            score_cls = nan_score
        else:
            score_cls = (2 * tp).to(torch.float) / denom

        scores[i-bg] += score_cls
    return reduce(scores, reduction=reduction)
