from collections import Sequence
from typing import Optional, Tuple

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


# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
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


# TODO:
def precision_recall_curve():
    pass


# TODO:
def multilabel_precision_recall_curve():
    pass


# TODO:
def auc():
    pass


# TODO:
def auroc():
    pass


def dice_coefficient():
    pass
