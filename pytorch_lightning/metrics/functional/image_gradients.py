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


def _image_gradients_validate(img: torch.Tensor) -> torch.Tensor:
    """ Validates whether img is a 4D torch Tensor """

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"The `img` expects a value of <torch.Tensor> type but got {type(img)}")
    if img.ndim != 4:
        raise RuntimeError(f"The `img` expects a 4D tensor but got {img.ndim}D tensor")


def _compute_image_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes image gradients (dy/dx) for a given image """

    batch_size, channels, height, width = img.shape

    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]

    shapey = [batch_size, channels, 1, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
    dy = dy.view(img.shape)

    shapex = [batch_size, channels, height, 1]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
    dx = dx.view(img.shape)

    return dy, dx


def image_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the `gradients <https://en.wikipedia.org/wiki/Image_gradient>`_ of a given image using finite difference

    Args:
        img: An ``(N, C, H, W)`` input tensor where C is the number of image channels

    Return:
        Tuple of (dy, dx) with each gradient of shape ``[N, C, H, W]``

    Example:
        >>> image = torch.arange(0, 1*1*5*5, dtype=torch.float32)
        >>> image = torch.reshape(image, (1, 1, 5, 5))
        >>> dy, dx = image_gradients(image)
        >>> dy[0, 0, :, :]
        tensor([[5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [0., 0., 0., 0., 0.]])

    .. note:: The implementation follows the 1-step finite difference method as followed
           by the TF implementation. The values are organized such that the gradient of
           [I(x+1, y)-[I(x, y)]] are at the (x, y) location
    """
    _image_gradients_validate(img)

    return _compute_image_gradients(img)
