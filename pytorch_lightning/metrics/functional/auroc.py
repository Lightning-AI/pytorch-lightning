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
from distutils.version import LooseVersion
from typing import Optional, Sequence, Tuple

import torch

from pytorch_lightning.metrics.classification.helpers import _input_format_classification, DataType
from pytorch_lightning.metrics.functional.auc import auc
from pytorch_lightning.metrics.functional.roc import roc
from pytorch_lightning.utilities import LightningEnum


class AverageMethods(LightningEnum):
    """ Type of averages """
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = None


def _auroc_update(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
    # use _input_format_classification for validating the input and get the mode of data
    _, _, mode = _input_format_classification(preds, target)

    if mode == 'multi class multi dim':
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.flatten()
    if mode == 'multi-label' and preds.ndim > 2:
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)

    return preds, target, mode


def _auroc_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = 'macro',
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
) -> torch.Tensor:
    # binary mode override num_classes
    if mode == 'binary':
        num_classes = 1

    # check max_fpr parameter
    if max_fpr is not None:
        if (not isinstance(max_fpr, float) and 0 < max_fpr <= 1):
            raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

        if LooseVersion(torch.__version__) < LooseVersion('1.6.0'):
            raise RuntimeError(
                "`max_fpr` argument requires `torch.bucketize` which"
                " is not available below PyTorch version 1.6"
            )

        # max_fpr parameter is only support for binary
        if mode != 'binary':
            raise ValueError(
                f"Partial AUC computation not available in"
                f" multilabel/multiclass setting, 'max_fpr' must be"
                f" set to `None`, received `{max_fpr}`."
            )

    # calculate fpr, tpr
    if mode == 'multi-label':
        # for multilabel we iteratively evaluate roc in a binary fashion
        output = [
            roc(preds[:, i], target[:, i], num_classes=1, pos_label=1, sample_weights=sample_weights)
            for i in range(num_classes)
        ]
        fpr = [o[0] for o in output]
        tpr = [o[1] for o in output]
    else:
        fpr, tpr, _ = roc(preds, target, num_classes, pos_label, sample_weights)

    # calculate standard roc auc score
    if max_fpr is None or max_fpr == 1:
        if num_classes != 1:
            # calculate auc scores per class
            auc_scores = [auc(x, y) for x, y in zip(fpr, tpr)]

            # calculate average
            if average == AverageMethods.NONE:
                return auc_scores
            elif average == AverageMethods.MACRO:
                return torch.mean(torch.stack(auc_scores))
            elif average == AverageMethods.WEIGHTED:
                if mode == DataType.MULTILABEL:
                    support = torch.sum(target, dim=0)
                else:
                    support = torch.bincount(target.flatten(), minlength=num_classes)
                return torch.sum(torch.stack(auc_scores) * support / support.sum())

            allowed_average = [e.value for e in AverageMethods]
            raise ValueError(
                f"Argument `average` expected to be one of the following:"
                f" {allowed_average} but got {average}"
            )

        return auc(fpr, tpr)

    max_fpr = torch.tensor(max_fpr, device=fpr.device)
    # Add a single point at max_fpr and interpolate its tpr value
    stop = torch.bucketize(max_fpr, fpr, out_int32=True, right=True)
    weight = (max_fpr - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr = torch.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = torch.cat([fpr[:stop], max_fpr.view(1)])

    # Compute partial AUC
    partial_auc = auc(fpr, tpr)

    # McClish correction: standardize result to be 0.5 if non-discriminant
    # and 1 if maximal
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def auroc(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = 'macro',
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None,
) -> torch.Tensor:
    """ Compute `Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Further_interpretations>`_

    Args:
        preds: predictions from model (logits or probabilities)
        target: Ground truth labels
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        average:
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range [0, max_fpr]. Should be a float between 0 and 1.
        sample_weight: sample weights for each data point

    Example (binary case):

        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> auroc(preds, target, pos_label=1)
        tensor(0.5000)

    Example (multiclass case):

        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> auroc(preds, target, num_classes=3)
        tensor(0.7778)
    """
    preds, target, mode = _auroc_update(preds, target)
    return _auroc_compute(preds, target, mode, num_classes, pos_label, average, max_fpr, sample_weights)
