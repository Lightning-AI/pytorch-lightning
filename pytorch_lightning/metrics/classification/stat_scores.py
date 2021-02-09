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
from typing import Any, Callable, Optional, Tuple

import torch

from pytorch_lightning.metrics.functional.stat_scores import _stat_scores_compute, _stat_scores_update
from pytorch_lightning.metrics.metric import Metric


class StatScores(Metric):
    """Computes the number of true positives, false positives, true negatives, false negatives.
    Related to `Type I and Type II errors <https://en.wikipedia.org/wiki/Type_I_and_type_II_errors>`__
    and the `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion>`__.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case.

    Accepts all inputs listed in :ref:`extensions/metrics:input types`.

    Args:
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0 or 1) predictions, in the case of binary or multi-label inputs.

        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. If this parameter is set for multi-label
            inputs, it will take precedence over ``threshold``. For (multi-dim) multi-class inputs,
            this parameter defaults to 1.

            Should be left unset (``None``) for inputs with label predictions.

        reduce:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all [sample, class]
              combinations (globally). Each statistic is represented by a single integer.
            - ``'macro'``: Counts the statistics for each class separately (over all samples).
              Each statistic is represented by a ``(C,)`` tensor. Requires ``num_classes``
              to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Each statistic is represented by a ``(N, )`` 1d tensor.

            Note that what is considered a sample in the multi-dimensional multi-class case
            depends on the value of ``mdmc_reduce``.

        num_classes:
            Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.

        ignore_index:
            Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.

        mdmc_reduce:
            Defines how the multi-dimensional multi-class inputs are handeled. Should be
            one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class (see :ref:`extensions/metrics:input types` for the definition of input types).

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then the outputs are concatenated together. In each
              sample the extra axes ``...`` are flattened to become the sub-sample axis, and
              statistics for each sample are computed by treating the sub-sample axis as the
              ``N`` axis for that sample.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs are
              flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``reduce`` parameter applies as usual.

        is_multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <extensions/metrics:using the is_multiclass parameter>`
            for a more detailed explanation and examples.

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather.

    Example:

        >>> from pytorch_lightning.metrics.classification import StatScores
        >>> preds  = torch.tensor([1, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> stat_scores = StatScores(reduce='macro', num_classes=3)
        >>> stat_scores(preds, target)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> stat_scores = StatScores(reduce='micro')
        >>> stat_scores(preds, target)
        tensor([2, 2, 6, 2, 4])

    """

    def __init__(
        self,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        reduce: str = "micro",
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        mdmc_reduce: Optional[str] = None,
        is_multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reduce = reduce
        self.mdmc_reduce = mdmc_reduce
        self.num_classes = num_classes
        self.threshold = threshold
        self.is_multiclass = is_multiclass
        self.ignore_index = ignore_index
        self.top_k = top_k

        if not 0 < threshold < 1:
            raise ValueError(f"The `threshold` should be a float in the (0,1) interval, got {threshold}")

        if reduce not in ["micro", "macro", "samples"]:
            raise ValueError(f"The `reduce` {reduce} is not valid.")

        if mdmc_reduce not in [None, "samplewise", "global"]:
            raise ValueError(f"The `mdmc_reduce` {mdmc_reduce} is not valid.")

        if reduce == "macro" and (not num_classes or num_classes < 1):
            raise ValueError("When you set `reduce` as 'macro', you have to provide the number of classes.")

        if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
            raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

        if mdmc_reduce != "samplewise" and reduce != "samples":
            if reduce == "micro":
                zeros_shape = []
            elif reduce == "macro":
                zeros_shape = (num_classes, )
            default, reduce_fn = lambda: torch.zeros(zeros_shape, dtype=torch.long), "sum"
        else:
            default, reduce_fn = lambda: [], None

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx=reduce_fn)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`extensions/metrics:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities or labels)
            target: Ground truth values
        """

        tp, fp, tn, fn = _stat_scores_update(
            preds,
            target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            is_multiclass=self.is_multiclass,
            ignore_index=self.ignore_index,
        )

        # Update states
        if self.reduce != "samples" and self.mdmc_reduce != "samplewise":
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
        else:
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)

    def _get_final_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs concatenation on the stat scores if neccesary,
        before passing them to a compute function.
        """

        if isinstance(self.tp, list):
            tp = torch.cat(self.tp)
            fp = torch.cat(self.fp)
            tn = torch.cat(self.tn)
            fn = torch.cat(self.fn)
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        return tp, fp, tn, fn

    def compute(self) -> torch.Tensor:
        """
        Computes the stat scores based on inputs passed in to ``update`` previously.

        Return:
            The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
            to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The
            shape depends on the ``reduce`` and ``mdmc_reduce`` (in case of multi-dimensional
            multi-class data) parameters:

            - If the data is not multi-dimensional multi-class, then

              - If ``reduce='micro'``, the shape will be ``(5, )``
              - If ``reduce='macro'``, the shape will be ``(C, 5)``,
                where ``C`` stands for the number of classes
              - If ``reduce='samples'``, the shape will be ``(N, 5)``, where ``N`` stands for
                the number of samples

            - If the data is multi-dimensional multi-class and ``mdmc_reduce='global'``, then

              - If ``reduce='micro'``, the shape will be ``(5, )``
              - If ``reduce='macro'``, the shape will be ``(C, 5)``
              - If ``reduce='samples'``, the shape will be ``(N*X, 5)``, where ``X`` stands for
                the product of sizes of all "extra" dimensions of the data (i.e. all dimensions
                except for ``C`` and ``N``)

            - If the data is multi-dimensional multi-class and ``mdmc_reduce='samplewise'``, then

              - If ``reduce='micro'``, the shape will be ``(N, 5)``
              - If ``reduce='macro'``, the shape will be ``(N, C, 5)``
              - If ``reduce='samples'``, the shape will be ``(N, X, 5)``

        """
        tp, fp, tn, fn = self._get_final_stats()
        return _stat_scores_compute(tp, fp, tn, fn)
