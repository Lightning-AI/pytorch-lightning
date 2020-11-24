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


def _fbeta_compute(
    tp: torch.Tensor,
    fp: torch.Tensor,
    tn: torch.Tensor,
    fn: torch.Tensor,
    beta: float,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int,
) -> torch.Tensor:
    return _reduce_scores(
        numerator=(1 + beta ** 2) * tp,
        denominator=(1 + beta ** 2) * tp + beta ** 2 * fn + fp,
        weights=tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


def fbeta_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    zero_division: int = 0,
) -> torch.Tensor:
    """Computes the `F-score <https://en.wikipedia.org/wiki/F-score>`_ .

    The metric computes weighted hamonic mean of recall and precision, where the square of
    ``beta`` is the weight on recall.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    In case where you need to ignore a class in computing the score, an ``ignore_index``
    parameter is availible.

    Args:
        preds: Predictions from model (probabilities or labels)
        target: Ground truth values
        beta:
            Determines the weight of recall in the harmonic mean. Default 1 (equal weight to precision
            and recall).
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
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
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
        zero_division:
            Score to use for classes/samples, whose score has 0 in the denominator. Has to be either
            0 [default] or 1.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:

        >>> from pytorch_lightning.metrics.functional import fbeta_score
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> fbeta_score(preds, target, beta=2.0, average='macro', num_classes=3)
        tensor(0.2778)
        >>> fbeta_score(preds, target, beta=2.0, average='micro')
        tensor(0.2500)

    """

    reduce = "macro" if average in ["weighted", "none", None] else average

    if zero_division not in [0, 1]:
        raise ValueError("zero_division has to be either 0 or 1")

    tp, fp, tn, fn = _stat_scores_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, is_multiclass, ignore_index
    )

    return _fbeta_compute(tp, fp, tn, fn, beta, average, mdmc_average, zero_division)


def f1_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    zero_division: int = 0,
) -> torch.Tensor:
    """Computes the `F1-score <https://en.wikipedia.org/wiki/F-score>`_ (also known as Dice score).

    The metric computes the hamonic mean of recall and precision. It is equivalent to
    :class:`~pytorch_lightning.metrics.classification.FBeta` with ``beta=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    In case where you need to ignore a class in computing the score, an ``ignore_index``
    parameter is availible.

    Args:
        preds: Predictions from model (probabilities or labels)
        target: Ground truth values
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
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
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
        zero_division:
            Score to use for classes/samples, whose score has 0 in the denominator. Has to be either
            0 [default] or 1.

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:

        >>> from pytorch_lightning.metrics.functional import f1_score
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> f1_score(preds, target, average='macro', num_classes=3)
        tensor(0.2222)
        >>> f1_score(preds, target, average='micro')
        tensor(0.2500)

    """

    reduce = "macro" if average in ["weighted", "none", None] else average

    if zero_division not in [0, 1]:
        raise ValueError("zero_division has to be either 0 or 1")

    tp, fp, tn, fn = _stat_scores_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, is_multiclass, ignore_index
    )

    return _fbeta_compute(tp, fp, tn, fn, 1.0, average, mdmc_average, zero_division)
