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
from typing import Optional

import torch
from pytorch_lightning.metrics.functional.reduction import _reduce_scores
from pytorch_lightning.metrics.functional.stat_scores import _stat_scores_update


def _iou_compute(
    tp: torch.Tensor,
    fp: torch.Tensor,
    tn: torch.Tensor,
    fn: torch.Tensor,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int,
) -> torch.Tensor:
    return _reduce_scores(
        numerator=tp,
        denominator=tp + fn + fp,
        weights=tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


def iou(
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    logits: bool = True,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    zero_division: int = 0,
) -> torch.Tensor:
    """Computes the Intersection over Union (the ratio ``tp / (tp + fp + fn)``), also known
    as the Jaccard Index or Jaccard Score).

    This metric can be used for image segmentation out of the box - you only need to
    properly set the ``average`` and ``mdmc_average`` parameters. In this case a single
    "sample" would correspond to an image.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    In case where you need to ignore a class in computing the score, an ``ignore_index``
    parameter is availible.

    Args:
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, by counting the statistics
              (tp, fp, tn, fn) accross all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics accross classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics accross classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            Note that what is considered a sample in the multi-dimensional multi-class case
            depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`metrics:Input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs (see :ref:`metrics:Input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.

        threshold:
            Threshold probability value for transforming probability/logit predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. If ``logits=True``,
            this value is transformed to logits by ``logit_t = ln(t / (1-t))``. Default: 0.5
        logits:
            If predictions are floats, whether they are probabilities or logits. Default ``True``
            (predictions are logits).
        is_multiclass:
            If ``False``, treat multi-class and multi-dim multi-class inputs with 1 or 2 classes as
            binary and multi-label, respectively. If ``True``, treat binary and multi-label inputs
            as multi-class or multi-dim multi-class with 2 classes, respectively.
            Defaults to ``None``, which treats inputs as they appear.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. Has no effect if given an int that
            is not in the range ``[0, C-1]``, or if  ``C=1``, where ``C`` is the number of classes.

            If an index is ignored, and ``average=None`` or ``'none'``, the score for the ignored class
            will be returned as ``nan`` (to not break the indexing of other labels).

    Return:
        The of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:

        >>> from pytorch_lightning.metrics.functional import iou
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> iou(preds, target, average='macro', num_classes=3)
        tensor(0.1667)
        >>> iou(preds, target, average='micro')
        tensor(0.1429)

    """

    reduce = "macro" if average in ["weighted", "none", None] else average

    if zero_division not in [0, 1]:
        raise ValueError("zero_division has to be either 0 or 1")

    # Check average
    if average not in ["micro", "macro", "weighted", "samples", "none", None]:
        raise ValueError("Uncrecognized average option: %s" % average)

    tp, fp, tn, fn = _stat_scores_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, logits, is_multiclass, ignore_index
    )

    return _iou_compute(tp, fp, tn, fn, average, mdmc_average, zero_division)
