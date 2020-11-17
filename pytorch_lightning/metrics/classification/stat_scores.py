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
from typing import Optional, Any, Tuple, Callable

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import to_onehot
from pytorch_lightning.metrics.classification.utils import _input_format_classification


def _stat_scores(
    preds: torch.Tensor, target: torch.Tensor, average: str = "micro"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    The shape of the returned tensors depnds on the shape of the inputs
    and the `average` parameter:

        * If inputs are of the shape (N, C), then

            * If average is 'micro', the returned tensors are 1 element tensors
            * If average is one of 'macro', 'weighted', 'none' or None, the returned
              tensors are (C,) 1d tensors
            * If average is 'samples, the returned tensors are 1d (N,) tensors

        * If inputs are of the shape (N, C, X), then

            * If average is 'micro', the returned tensors are (N,) 1d tensors
            * If average is one of 'macro', 'weighted', 'none' or None, the returned
              tensors are (N,C) 2d tensors
            * If average is 'samples, the returned tensors are 1d (N,X) 2d tensors

    Parameters
    ----------
    labels
        An (N, C) or (N, C, X) tensor of true labels (0 or 1)
    preds
        An (N, C) or (N, C, X) tensor of predictions (0 or 1)
    average
        One of 'micro', 'macro', 'weighted', 'samples' or 'none' (None)

    Returns
    -------
    tp, fp, tn, fn
    """
    is_multidim = len(preds.shape) == 3

    if average in ["micro"]:
        dim = [0, 1] if not is_multidim else [1, 2]
    elif average in ["macro", "weighted", "none", None]:
        dim = 0 if not is_multidim else 2
    elif average in ["samples"]:
        dim = 1

    true_pred, false_pred = target == preds, target != preds

    tp = (true_pred * (preds == 1)).sum(dim=dim)
    fp = (false_pred * (preds == 1)).sum(dim=dim)

    tn = (true_pred * (preds == 0)).sum(dim=dim)
    fn = (false_pred * (preds == 0)).sum(dim=dim)

    return tp.int(), fp.int(), tn.int(), fn.int()


def _reduce_scores(scores: torch.Tensor, weights: torch.Tensor, average: str) -> torch.Tensor:
    """Reduce scores according to the average method.

    Parameters
    ----------
    scores
        A tensor with the scores to be reduced
    weights
        If average='weighted', a tensor of weights - should be non-negative
    average
        One of 'micro', 'macro', 'weighted', 'samples' or 'none' (None)
    """

    if average in ["micro", "none", None]:
        return scores
    elif average in ["macro", "samples"]:
        return scores.mean()
    elif average == "weighted":
        w_scores = scores * (weights / weights.sum())
        return w_scores.sum()


class StatScores(Metric):
    """Computes the number of true positives, false positives, true negatives, false negatives.

    The reduction method is controlled by the ``average`` parameter, and the
    ``mdmc_average`` parameter in the multi-dimensional multi-class case. Accepts
    all inputs listed in :ref:`metrics:Input types`.

    This metric is a good choice for subclassing for other metrics based on tp, fp, tn, fn
    statistics (such as :class:`~pytorch_lightning.metrics.classification.Recall` and
    :class:`~pytorch_lightning.metrics.classification.Precision`), as it already implements
    the logic for computing the values, so only the ``update`` function where the final
    calculations are performed should be subclassed.

    Args:
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all [sample, class]
              combinations (globally). Produces a one element tensor for each statistic.
            - ``'macro'``, ``'weighted'``, ``'none'`` or ``None``: Counts the statistics
              for each class separately (over all samples). Produces a ``(C, )`` 1d tensor.
              Requires ``num_classes`` to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Produces a ``(N, )`` 1d tensor.

            Note that what is considered a sample in the multi-dimensional multi-class case
            depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how the multi-dimensional multi-class inputs are handeled. Should be
            one of the following:

            - ``None``: This is the default value and should be left unchanged if your data
              is not multi-dimensional multi-class.

            - ``'samples'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis. This is done by, for each sample, treating the flattened
              extra axes ``...`` (see :ref:`metrics:Input types`) as the ``N`` dimension within
              the sample, and computing the statistics for the sample based on that.

              For each statistic the final product are the concatenated values of the statistics
              for each sample. This depends on the value of the value ``average`` parameter: if
              it equals ``micro``, then this is a ``(N, )`` 1d tensor, if it equals ``macro`` or
              equivalent than this is a ``(N, C)`` 2d tensor, and if it equals ``samples``, than
              this is a ``(N, X)`` 2d tensor, where ``X`` is the size of the flattened ``...``
              dimensions.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs (see :ref:`metrics:Input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.

        threshold:
            Threshold probability value for binary or multi-label logits/probabilities. default: 0.5
        logits:
            If predictions are floats, whether the values passed in are probabilities or logits.
            This information is used to know how to transform the ``threshold`` probability before
            comparison.
        is_multiclass: if ``True``, treat binary and multi-label inputs as multi-class or multi-dim
            multi-class with 2 classes, respectively. If ``False``, treat multi-class and multi-dim
            multi-class inputs with 1 or 2 classes as binary and multi-label, respectively.
            Defaults to ``None``, which treats inputs as they appear.

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
        >>> stat_scores = StatScores(average='macro', num_classes=3)
        >>> stat_scores(preds, target)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]], dtype=torch.int32)
        >>> stat_scores = StatScores(average='micro')
        >>> stat_scores(preds, target)
        tensor([2, 2, 6, 2, 4], dtype=torch.int32)

    """

    def __init__(
        self,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        logits: bool = False,
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

        self.average = average
        self.mdmc_average = mdmc_average
        self.num_classes = num_classes
        self.threshold = threshold
        self.logits = logits
        self.is_multiclass = is_multiclass

        if average not in ["micro", "macro", "weighted", "none", None, "samples"]:
            raise ValueError("average %s is not valid." % average)

        if mdmc_average not in [None, "samples", "global"]:
            raise ValueError("mdmc_average %s is not valid." % mdmc_average)

        if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
            raise ValueError(
                "When you set the average as macro, weighted or none, you have to provide the number of classes."
            )

        if mdmc_average != 'samples':
            if average in ["micro"]:
                default, reduce_fn = torch.tensor(0), "sum"
            elif average in ["macro", "weighted", "none", None]:
                default, reduce_fn = torch.zeros((num_classes,), dtype=torch.int), "sum"
            elif average == "samples":
                default, reduce_fn = torch.empty(0), "cat"
        else:
            default, reduce_fn = torch.empty(0), "cat"

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default.detach().clone(), dist_reduce_fx=reduce_fn)

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        preds, target, _ = _input_format_classification(
            preds,
            target,
            threshold=self.threshold,
            num_classes=self.num_classes,
            logits=self.logits,
            is_multiclass=self.is_multiclass,
        )

        if len(preds.shape) == 3:
            if not self.mdmc_average:
                raise ValueError(
                    "When your inputs are multi-dimensional multi-class,"
                    "you have to set mdmc_average to either 'samples' or 'global'"
                )
            elif self.mdmc_average == "global":
                preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
                target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        tp, fp, tn, fn = _stat_scores(preds, target, average=self.average)

        if self.average in ["micro", "macro", "weighted", "none", None] and self.mdmc_average != "samples":
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

        else:
            if isinstance(self.tp, list):
                self.tp, self.fp, self.tn, self.fn = tp, fp, tn, fn
            else:
                self.tp = torch.cat((self.tp, tp))
                self.fp = torch.cat((self.fp, fp))
                self.tn = torch.cat((self.tn, tn))
                self.fn = torch.cat((self.fn, fn))

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = [
            self.tp.unsqueeze(-1),
            self.fp.unsqueeze(-1),
            self.tn.unsqueeze(-1),
            self.fn.unsqueeze(-1),
            self.tp.unsqueeze(-1) + self.fn.unsqueeze(-1),
        ]

        return torch.cat(outputs, -1).int()
