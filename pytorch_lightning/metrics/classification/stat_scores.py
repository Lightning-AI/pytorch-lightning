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
from pytorch_lightning.metrics.utils import dim_zero_cat
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.stat_scores import _stat_scores_update, _stat_scores_compute


def _dim_zero_cat_and_put_back(tensor: torch.Tensor):
    """ Needed as we don't need the process dimension in sync reduce """

    out = dim_zero_cat(tensor)
    out = out.reshape(-1, *out.shape[2:])

    return out


class StatScores(Metric):
    """Computes the number of true positives, false positives, true negatives, false negatives.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    Args:
        reduce:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all [sample, class]
              combinations (globally). Produces a one element tensor for each statistic.
            - ``'macro'``: Counts the statistics for each class separately (over all samples).
              Produces a ``(C, )`` 1d tensor. Requires ``num_classes`` to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Produces a ``(N, )`` 1d tensor.

            Note that what is considered a sample in the multi-dimensional multi-class case
            depends on the value of ``mdmc_reduce``.

        mdmc_reduce:
            Defines how the multi-dimensional multi-class inputs are handeled. Should be
            one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then concatenating the outputs together. This is
              done by, for each sample, treating the flattened extra axes ``...`` (see
              :ref:`metrics:Input types`) as the ``N`` dimension within the sample, and computing
              the statistics for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs (see :ref:`metrics:Input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``reduce`` parameter applies as usual.

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

            If an index is ignored, and ``reduce='macro'``, the class statistics for the ignored
            class will all be returned as ``nan`` (to not break the indexing of other labels).
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
        reduce: str = "micro",
        mdmc_reduce: Optional[str] = None,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        logits: bool = True,
        is_multiclass: Optional[bool] = None,
        ignore_index: Optional[int] = None,
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
        self.logits = logits
        self.is_multiclass = is_multiclass
        self.ignore_index = ignore_index

        if reduce not in ["micro", "macro", "samples"]:
            raise ValueError("reduce %s is not valid." % reduce)

        if mdmc_reduce not in [None, "samplewise", "global"]:
            raise ValueError("mdmc_reduce %s is not valid." % mdmc_reduce)

        if reduce == "macro" and (not num_classes or num_classes < 1):
            raise ValueError("When you set reduce as macro, you have to provide the number of classes.")

        if mdmc_reduce != "samplewise":
            if reduce == "micro":
                default, reduce_fn = torch.tensor(0), "sum"
            elif reduce == "macro":
                default, reduce_fn = torch.zeros((num_classes,), dtype=torch.int), "sum"
            elif reduce == "samples":
                default, reduce_fn = torch.empty(0), _dim_zero_cat_and_put_back
        else:
            default, reduce_fn = torch.empty(0), _dim_zero_cat_and_put_back

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default.detach().clone(), dist_reduce_fx=reduce_fn)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`metrics:Input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, logits, or labels)
            target: Ground truth values
        """

        tp, fp, tn, fn = _stat_scores_update(
            preds,
            target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            logits=self.logits,
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
            self.tp = torch.cat((self.tp, tp))
            self.fp = torch.cat((self.fp, fp))
            self.tn = torch.cat((self.tn, tn))
            self.fn = torch.cat((self.fn, fn))

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

        return _stat_scores_compute(self.tp, self.fp, self.tn, self.fn)
