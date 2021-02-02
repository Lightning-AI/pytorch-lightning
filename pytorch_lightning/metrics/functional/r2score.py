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
from typing import Tuple

import torch

from pytorch_lightning.metrics.utils import _check_same_shape
from pytorch_lightning.utilities import rank_zero_warn


def _r2score_update(
    preds: torch.tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_same_shape(preds, target)
    if preds.ndim > 2:
        raise ValueError(
            'Expected both prediction and target to be 1D or 2D tensors,'
            f' but recevied tensors with dimension {preds.shape}'
        )
    if len(preds) < 2:
        raise ValueError('Needs atleast two samples to calculate r2 score.')

    sum_error = torch.sum(target, dim=0)
    sum_squared_error = torch.sum(torch.pow(target, 2.0), dim=0)
    residual = torch.sum(torch.pow(target - preds, 2.0), dim=0)
    total = target.size(0)

    return sum_squared_error, sum_error, residual, total


def _r2score_compute(
    sum_squared_error: torch.Tensor,
    sum_error: torch.Tensor,
    residual: torch.Tensor,
    total: torch.Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average"
) -> torch.Tensor:
    mean_error = sum_error / total
    diff = sum_squared_error - sum_error * mean_error
    raw_scores = 1 - (residual / diff)

    if multioutput == "raw_values":
        r2score = raw_scores
    elif multioutput == "uniform_average":
        r2score = torch.mean(raw_scores)
    elif multioutput == "variance_weighted":
        diff_sum = torch.sum(diff)
        r2score = torch.sum(diff / diff_sum * raw_scores)
    else:
        raise ValueError(
            'Argument `multioutput` must be either `raw_values`,'
            f' `uniform_average` or `variance_weighted`. Received {multioutput}.'
        )

    if adjusted < 0 or not isinstance(adjusted, int):
        raise ValueError('`adjusted` parameter should be an integer larger or' ' equal to 0.')

    if adjusted != 0:
        if adjusted > total - 1:
            rank_zero_warn(
                "More independent regressions than datapoints in"
                " adjusted r2 score. Falls back to standard r2 score.", UserWarning
            )
        elif adjusted == total - 1:
            rank_zero_warn("Division by zero in adjusted r2 score. Falls back to" " standard r2 score.", UserWarning)
        else:
            r2score = 1 - (1 - r2score) * (total - 1) / (total - adjusted - 1)
    return r2score


def r2score(
    preds: torch.Tensor,
    target: torch.Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> torch.Tensor:
    r"""
    Computes r2 score also known as `coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

    .. math:: R^2 = 1 - \frac{SS_res}{SS_tot}

    where :math:`SS_res=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_tot=\sum_i (y_i - \bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_adj = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.

    Args:
        preds: estimated labels
        target: ground truth labels
        adjusted: number of independent regressors for calculating adjusted r2 score.
            Default 0 (standard r2 score).
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is ``'uniform_average'``.):

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Example:

        >>> from pytorch_lightning.metrics.functional import r2score
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> r2score(preds, target)
        tensor(0.9486)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2score(preds, target, multioutput='raw_values')
        tensor([0.9654, 0.9082])
    """
    sum_squared_error, sum_error, residual, total = _r2score_update(preds, target)
    return _r2score_compute(sum_squared_error, sum_error, residual, total, adjusted, multioutput)
