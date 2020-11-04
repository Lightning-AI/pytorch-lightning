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
from typing import Optional, Sequence, Tuple, List, Union

import torch
import torch.nn.functional as F

from pytorch_lightning.utilities import rank_zero_warn


def _binary_clf_curve(
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[Sequence] = None,
        pos_label: int = 1.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
    """
    if sample_weights is not None and not isinstance(sample_weights, torch.Tensor):
        sample_weights = torch.tensor(sample_weights, device=preds.device, dtype=torch.float)

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]
    desc_score_indices = torch.argsort(preds, descending=True)

    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    if sample_weights is not None:
        weight = sample_weights[desc_score_indices]
    else:
        weight = 1.

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, (0, 1), value=target.size(0) - 1)
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    if sample_weights is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, preds[threshold_idxs]


def _precision_recall_curve_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )
    # single class evaluation
    if len(preds.shape) == len(target.shape):
        if num_classes is not None and num_classes != 1:
            raise ValueError('Preds and target have equal shape, but number of classes is different from 1')
        num_classes = 1
        if pos_label is None:
            rank_zero_warn('`pos_label` automatically set 1.')
            pos_label = 1
        preds = preds.flatten()
        target = target.flatten()

    # multi class evaluation
    if len(preds.shape) == len(target.shape) + 1:
        if pos_label is not None:
            rank_zero_warn('Argument `pos_label` should be `None` when running'
                           f'multiclass precision recall curve. Got {pos_label}')
        if num_classes != preds.shape[1]:
            raise ValueError(f'Argument `num_classes` was set to {num_classes} in'
                             f'metric `precision_recall_curve` but detected {preds.shape[1]}'
                             'number of classes from predictions')
        preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1)
        target = target.flatten()

    return preds, target, num_classes, pos_label


def _precision_recall_curve_compute(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        pos_label: int,
        sample_weights: Optional[Sequence] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:

    if num_classes == 1:
        fps, tps, thresholds = _binary_clf_curve(
            preds=preds,
            target=target,
            sample_weights=sample_weights,
            pos_label=pos_label
        )

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

    # Recursively call per class
    pr_curves = []
    for c in range(num_classes):
        preds_c = preds[:, c]
        res = precision_recall_curve(
            preds=preds_c,
            target=target,
            num_classes=1,
            pos_label=c,
            sample_weights=sample_weights,
        )
        pr_curves.append(res)

    return pr_curves


def precision_recall_curve(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        sample_weights: Optional[Sequence] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    preds, target, num_classes, pos_label = _precision_recall_curve_update(preds, target,
                                                                           num_classes, pos_label)
    return _precision_recall_curve_compute(preds, target, num_classes, pos_label, sample_weights)
