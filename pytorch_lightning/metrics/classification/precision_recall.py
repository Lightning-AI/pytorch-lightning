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
from typing import Optional, Any, Callable

import torch
from pytorch_lightning.metrics.classification.utils import _mask_zeros
from pytorch_lightning.metrics.classification.stat_scores import StatScores, _reduce_scores


class Precision(StatScores):
    """Computes the precision score (the ratio ``tp / (tp + fp)``).

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    In case where you need to ignore a class in computing the score, a ``ignore_index``
    parameter is availible.

    The of the returned tensor depends on the ``average`` parameter:

    - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
    - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
    of classes

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
            will be returned as ``-1`` (to not break the indexing of other labels).
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

    Example:

        >>> from pytorch_lightning.metrics.classification import Precision
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> stat_scores = Precision(average='macro', num_classes=3)
        >>> stat_scores(preds, target)
        tensor(0.1667)
        >>> stat_scores = Precision(average='micro')
        >>> stat_scores(preds, target)
        tensor(0.2500)

    """

    def __init__(
        self,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        logits: bool = True,
        is_multiclass: Optional[bool] = None,
        ignore_index: Optional[int] = None,
        zero_division: int = 0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            reduce="macro" if average in ["weighted", "none", None] else average,
            mdmc_average=mdmc_average,
            threshold=threshold,
            num_classes=num_classes,
            logits=logits,
            is_multiclass=is_multiclass,
            ignore_index=ignore_index,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if zero_division not in [0, 1]:
            raise ValueError("zero_division has to be either 0 or 1")

        # Check average

        self.zero_division = zero_division
        self.average = average

    def compute(self):
        """
        Computes the precision score based on inputs passed in to ``update`` previously.
        """
        tp = self.tp.float()
        total_pos_pred = _mask_zeros((self.tp + self.fp).float())
        prec_scores = tp / total_pos_pred

        return _reduce_scores(prec_scores, self.tp + self.fn, self.average)


class Recall(StatScores):
    def compute(self):
        tp = self.tp.float()
        total_pos = _mask_zeros((self.tp + self.fn).float())
        rec_scores = tp / total_pos

        return _reduce_scores(rec_scores, self.tp + self.fn, self.average)


class FBeta(StatScores):
    def __init__(
        self,
        beta: float = 1.0,
        average: str = "micro",
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        logits: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            threshold=threshold,
            num_classes=num_classes,
            logits=logits,
            average=average,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.beta = beta

    def compute(self):
        numerator = (1 + self.beta ** 2) * self.tp.float()
        denominator = numerator + self.beta ** 2 * self.fn.float() + self.fp.float()

        return _reduce_scores(numerator / denominator, self.tp + self.fn, self.average)
