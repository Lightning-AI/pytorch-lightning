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
import torch

from typing import Any, Callable, Optional, Union


METRIC_EPS = 1e-6


def dim_zero_cat(x):
    return torch.cat(x, dim=0)


def dim_zero_sum(x):
    return torch.sum(x, dim=0)


def dim_zero_mean(x):
    return torch.mean(x, dim=0)


def _flatten(x):
    return [item for sublist in x for item in sublist]


def to_onehot(
        tensor: torch.Tensor,
        num_classes: int,
) -> torch.Tensor:
    """
    Converts a dense label tensor to one-hot format

    Args:
        tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x, num_classes=4)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], num_classes, *shape[1:],
                                dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def _check_same_shape(pred: torch.Tensor, target: torch.Tensor):
    """ Check that predictions and target have the same shape, else raise error """
    if pred.shape != target.shape:
        raise RuntimeError('Predictions and targets are expected to have the same shape')
