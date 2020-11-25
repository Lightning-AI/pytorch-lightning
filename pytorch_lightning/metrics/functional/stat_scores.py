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
from typing import Tuple, Union, Optional

import torch
from pytorch_lightning.metrics.classification.utils import _input_format_classification


def _del_column(tensor: torch.Tensor, index: int):
    """ Delete the column at index."""

    return torch.cat([tensor[:, :index], tensor[:, (index + 1) :]], 1)


def _stat_scores(
    preds: torch.Tensor, target: torch.Tensor, reduce: str = "micro"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    The shape of the returned tensors depnds on the shape of the inputs
    and the `reduce` parameter:

        * If inputs are of the shape (N, C), then

            * If reduce is 'micro', the returned tensors are 1 element tensors
            * If reduce is one of 'macro', 'weighted', 'none' or None, the returned
              tensors are (C,) 1d tensors
            * If reduce is 'samples, the returned tensors are 1d (N,) tensors

        * If inputs are of the shape (N, C, X), then

            * If reduce is 'micro', the returned tensors are (N,) 1d tensors
            * If reduce is one of 'macro', 'weighted', 'none' or None, the returned
              tensors are (N,C) 2d tensors
            * If reduce is 'samples, the returned tensors are 1d (N,X) 2d tensors

    Parameters
    ----------
    labels
        An (N, C) or (N, C, X) tensor of true labels (0 or 1)
    preds
        An (N, C) or (N, C, X) tensor of predictions (0 or 1)
    reduce
        One of 'micro', 'macro', 'samples'

    Returns
    -------
    tp, fp, tn, fn
    """
    is_multidim = len(preds.shape) == 3

    if reduce == "micro":
        dim = [0, 1] if not is_multidim else [1, 2]
    elif reduce == "macro":
        dim = 0 if not is_multidim else 2
    elif reduce == "samples":
        dim = 1

    true_pred, false_pred = target == preds, target != preds

    tp = (true_pred * (preds == 1)).sum(dim=dim)
    fp = (false_pred * (preds == 1)).sum(dim=dim)

    tn = (true_pred * (preds == 0)).sum(dim=dim)
    fn = (false_pred * (preds == 0)).sum(dim=dim)

    return tp.int(), fp.int(), tn.int(), fn.int()


def _stat_scores_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "micro",
    mdmc_reduce: Optional[str] = None,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    preds, target, _ = _input_format_classification(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
    )

    if len(preds.shape) == 3:
        if not mdmc_reduce:
            raise ValueError(
                "When your inputs are multi-dimensional multi-class,"
                "you have to set mdmc_reduce to either 'samplewise' or 'global'"
            )
        if mdmc_reduce == "global":
            shape_permute = list(range(preds.ndim))
            shape_permute[1] = shape_permute[-1]
            shape_permute[2:] = range(1, len(shape_permute) - 1)

            preds = preds.permute(*shape_permute).reshape(-1, preds.shape[1])
            target = target.permute(*shape_permute).reshape(-1, target.shape[1])

    # Delete what is in ignore_index, if applicable (and classes don't matter):
    if ignore_index and reduce in ["micro", "samples"] and preds.shape[1] > 1:
        if 0 <= ignore_index < preds.shape[1]:
            preds = _del_column(preds, ignore_index)
            target = _del_column(target, ignore_index)

    tp, fp, tn, fn = _stat_scores(preds, target, reduce=reduce)

    # Take care of ignore_index
    if ignore_index and reduce == "macro":
        if num_classes > 1 and 0 <= ignore_index < num_classes:
            if mdmc_reduce == "global" or not mdmc_reduce:
                tp[ignore_index] = -1
                fp[ignore_index] = -1
                tn[ignore_index] = -1
                fn[ignore_index] = -1
            else:
                tp[:, ignore_index] = -1
                fp[:, ignore_index] = -1
                tn[:, ignore_index] = -1
                fn[:, ignore_index] = -1

    return tp, fp, tn, fn


def _stat_scores_compute(tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:

    outputs = [
        tp.unsqueeze(-1),
        fp.unsqueeze(-1),
        tn.unsqueeze(-1),
        fn.unsqueeze(-1),
        tp.unsqueeze(-1) + fn.unsqueeze(-1),  # support
    ]
    outputs = torch.cat(outputs, -1).long()

    # To standardzie ignore_index statistics as -1
    outputs = torch.where(outputs < 0, torch.tensor(-1, device=outputs.device), outputs)

    return outputs


def stat_scores(
    preds: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "micro",
    mdmc_reduce: Optional[str] = None,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Computes the number of true positives, false positives, true negatives, false negatives.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`metrics:Input types`.

    Args:
        preds: Predictions from model (probabilities, or labels)
        target: Ground truth values
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

            If an index is ignored, and ``reduce='macro'``, the class statistics for the ignored
            class will all be returned as ``nan`` (to not break the indexing of other labels).

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

    Example:

        >>> from pytorch_lightning.metrics.functional import stat_scores
        >>> preds  = torch.tensor([1, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, reduce='macro', num_classes=3)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> stat_scores(preds, target, reduce='micro')
        tensor([2, 2, 6, 2, 4])

    """

    if reduce not in ["micro", "macro", "samples"]:
        raise ValueError("reduce %s is not valid." % reduce)

    if mdmc_reduce not in [None, "samplewise", "global"]:
        raise ValueError("mdmc_reduce %s is not valid." % mdmc_reduce)

    if reduce == "macro" and (not num_classes or num_classes < 1):
        raise ValueError("When you set reduce as macro, you have to provide the number of classes.")

    tp, fp, tn, fn = _stat_scores_update(
        preds, target, reduce, mdmc_reduce, threshold, num_classes, is_multiclass, ignore_index
    )
    return _stat_scores_compute(tp, fp, tn, fn)
