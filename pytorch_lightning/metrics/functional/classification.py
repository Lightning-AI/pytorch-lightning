from collections import Sequence
from typing import Optional, Tuple, Callable

import torch

from pytorch_lightning.metrics.functional.reduction import reduce


def to_onehot(tensor: torch.Tensor,
              n_classes: Optional[int] = None) -> torch.Tensor:
    if n_classes is None:
        n_classes = int(tensor.max().detach().item() + 1)
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], n_classes, *shape[2:],
                                dtype=dtype, device=device)
    return tensor_onehot.scatter_(1, tensor, 1.0)


def to_categorical(tensor: torch.Tensor, argmax_dim: int = 1) -> torch.Tensor:
    return torch.argmax(tensor, dim=argmax_dim)


def get_num_classes(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: Optional[int]) -> int:
    if num_classes is None:
        if pred.ndim > target.ndim:
            num_classes = pred.size(1)
        else:
            num_classes = target.max().detach().item()
    return num_classes


def stat_scores(pred: torch.Tensor, target: torch.Tensor,
                class_index: int, argmax_dim: int = 1
                ) -> Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((pred == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((pred != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((pred != class_index) * (target == class_index)).to(torch.long).sum()

    return tp, fp, tn, fn


def stat_scores_multiple_classes(pred: torch.Tensor, target: torch.Tensor,
                                 num_classes: Optional[int] = None,
                                 argmax_dim: int = 1
                                 ) -> Tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]:
    num_classes = get_num_classes(pred=pred, target=target,
                                  num_classes=num_classes)

    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tps = torch.zeros(num_classes, device=pred.device)
    fps = torch.zeros(num_classes, device=pred.device)
    tns = torch.zeros(num_classes, device=pred.device)
    fns = torch.zeros(num_classes, device=pred.device)

    for c in range(num_classes):
        tps[c], fps[c], tns[c], fns[c] = stat_scores(pred=pred, target=target,
                                                     class_index=c)

    return tps, fps, tns, fns


def accuracy(pred: torch.Tensor, target: torch.Tensor,
             num_classes: Optional[int] = None,
             reduction='elementwise_mean') -> torch.Tensor:
    tps, _, _, _ = stat_scores_multiple_classes(pred=pred, target=target,
                                                num_classes=num_classes)

    accuracies = tps / target.size(0)

    return reduce(accuracies, reduction=reduction)


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor,
                     normalize: bool = False) -> torch.Tensor:
    num_classes = get_num_classes(pred, target, None)

    d = target.size(-1)
    batch_vec = torch.arange(target.size(-1))
    # this will account for multilabel
    unique_labels = batch_vec * num_classes ** 2 + target * num_classes + pred

    bins = torch.bincount(unique_labels, minlength=d * num_classes ** 2)
    cm = bins.reshape(d, num_classes, num_classes).squeeze()

    if normalize:
        cm = cm / target.size(0)

    return cm


def precision_recall(pred: torch.Tensor, target: torch.Tensor,
                     num_classes: Optional[int] = None,
                     reduction: str = 'elementwise_mean'
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
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


def precision(pred: torch.Tensor, target: torch.Tensor,
              num_classes: Optional[int] = None,
              reduction: str = 'elementwise_mean') -> torch.Tensor:
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, reduction=reduction)[0]


def recall(pred: torch.Tensor, target: torch.Tensor,
           num_classes: Optional[int] = None,
           reduction: str = 'elementwise_mean') -> torch.Tensor:
    return precision_recall(pred=pred, target=target,
                            num_classes=num_classes, reduction=reduction)[1]


def fbeta_score(pred: torch.Tensor, target: torch.Tensor, beta: float,
                num_classes: Optional[int] = None,
                reduction: str = 'elementwise_mean') -> torch.Tensor:
    prec, rec = precision_recall(pred=pred, target=target,
                                 num_classes=num_classes,
                                 reduction='none')

    nom = (1 + beta ** 2) * prec * rec
    denom = ((beta ** 2) * prec + rec)
    fbeta = nom / denom

    return reduce(fbeta, reduction=reduction)


def f1_score(pred: torch.Tensor, target: torch.Tensor,
             num_classes: Optional[int] = None,
             reduction='elementwise_mean') -> torch.Tensor:
    return fbeta_score(pred=pred, target=target, beta=1.,
                       num_classes=num_classes, reduction=reduction)


def _binary_clf_curve(pred: torch.Tensor, target: torch.Tensor,
                      sample_weight: Optional[Sequence] = None,
                      pos_label: int = 1.) -> Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
    if sample_weight is not None and not isinstance(sample_weight,
                                                    torch.Tensor):
        sample_weight = torch.tensor(sample_weight, device=pred.device,
                                     dtype=torch.float)

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
        fps = torch.cumsum((1 - target) * weight)[threshold_idxs]

    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, pred[threshold_idxs]


def roc(pred: torch.Tensor, target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.) -> Tuple[torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor]:
    tps, fps, thresholds = _binary_clf_curve(pred=pred, target=target,
                                             sample_weight=sample_weight,
                                             pos_label=pos_label)

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
    fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
    thresholds = torch.cat([thresholds[0][None], thresholds])

    if fps[-1] <= 0:
        raise ValueError("No negative samples in targets, "
                         "false positive value should be meaningless")

    fpr = fps / fps[-1]

    if tps[-1] <= 0:
        raise ValueError("No positive samples in targets, "
                         "true positive value should be meaningless")

    tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def multiclass_roc(pred: torch.Tensor, target: torch.Tensor,
                   sample_weight: Optional[Sequence] = None,
                   num_classes: Optional[int] = None,
                   ) -> Tuple[Tuple[torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor]]:
    num_classes = get_num_classes(pred, target, num_classes)

    class_roc_vals = []

    for c in range(num_classes):
        pred_c = pred[:, c]

        class_roc_vals.append(roc(pred=pred_c, target=target,
                                  sample_weight=sample_weight, pos_label=c))

    return tuple(class_roc_vals)


def precision_recall_curve(pred: torch.Tensor,
                           target: torch.Tensor,
                           sample_weight: Optional[Sequence] = None,
                           pos_label: int = 1.) -> Tuple[torch.Tensor,
                                                         torch.Tensor,
                                                         torch.Tensor]:
    fps, tps, thresholds = _binary_clf_curve(pred=pred, target=target,
                                             sample_weight=sample_weight,
                                             pos_label=pos_label)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    sl = slice(0, last_ind.item())

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides thet are not yet supported in pytorch
    precision = torch.cat([reversed(precision[sl]),
                           torch.ones(1, dtype=precision.dtype,
                                      device=precision.device)])

    recall = torch.cat([reversed(recall[sl]),
                        torch.zeros(1, dtype=recall.dtype,
                                    device=recall.device)])

    thresholds = reversed(thresholds[sl])

    return precision, recall, thresholds


def multiclass_precision_recall_curve(pred: torch.Tensor, target: torch.Tensor,
                                      sample_weight: Optional[Sequence] = None,
                                      num_classes: Optional[int] = None,
                                      ) -> Tuple[Tuple[torch.Tensor,
                                                       torch.Tensor,
                                                       torch.Tensor]]:
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
    direction = 1.

    if reorder:
        # can't use lexsort here since it is not implemented for torch
        order = torch.argsort(x)
        x, y = x[order], y[order]
    else:
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx < 0).all():
                direction = -1.
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    return direction * torch.trapz(y, x)


def auc_decorator(reorder: bool = False) -> Callable:
    def wrapper(func_to_decorate: Callable) -> Callable:
        def new_func(*args, **kwargs) -> torch.Tensor:
            x, y = func_to_decorate(*args, **kwargs)[:2]

            return auc(x, y, reorder=reorder)

        return new_func

    return wrapper


def multiclass_auc_decorator(reorder: bool = False) -> Callable:
    def wrapper(func_to_decorate: Callable) -> Callable:
        def new_func(*args, **kwargs) -> torch.Tensor:
            results = []
            for class_result in func_to_decorate(*args, **kwargs):
                x, y = class_result[:2]
                results.append(auc(x, y, reorder=reorder))

            return torch.cat(results)

        return new_func

    return wrapper


@auc_decorator(reorder=False)
def auroc(pred: torch.Tensor, target: torch.Tensor,
          sample_weight: Optional[Sequence] = None,
          pos_label: int = 1.) -> torch.Tensor:
    return roc(pred=pred, target=target, sample_weight=sample_weight,
               pos_label=pos_label)


@auc_decorator(reorder=False)
def average_precision(pred: torch.Tensor, target: torch.Tensor,
                      sample_weight: Optional[Sequence] = None,
                      pos_label: int = 1.) -> torch.Tensor:
    return precision_recall_curve(pred=pred, target=target,
                                  sample_weight=sample_weight,
                                  pos_label=pos_label)


def dice_score(pred: torch.Tensor, target: torch.Tensor, bg: bool = False,
               nan_score: float = 0.0, no_fg_score: float = 0.0,
               reduction: str = 'elementwise_mean'):
    n_classes = pred.shape[1]
    bg = (1 - int(bool(bg)))
    scores = torch.zeros(n_classes - bg, device=pred.device, dtype=pred.dtype)
    for i in range(bg, n_classes):
        if not (target == i).any():
            # no foreground class
            scores[i] += no_fg_score
            continue

        tp, fp, tn, fn = stat_scores(pred=pred[:, i], target=target,
                                     class_index=i)

        denom = (2 * tp + fp + fn).to(torch.float)

        if torch.isclose(denom, torch.zeros_like(denom)).any():
            # nan result
            score_cls = nan_score
        else:
            score_cls = (2 * tp).to(torch.float) / denom

        scores[i] += score_cls
    return reduce(scores, reduction=reduction)
