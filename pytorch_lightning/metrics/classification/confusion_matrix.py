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
from typing import Any, Optional, Callable

import torch

from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional.confusion_matrix import _confusion_matrix_update, _confusion_matrix_compute


class ConfusionMatrix(Metric):
    """
    Computes the confusion matrix.

    While this metric is mainly meant for multi-class inputs, it accepts all input types
    listed in :ref:`metrics:Input types`. If you pass binary or  multi-label inputs it
    will convert them to 2-class multi-class or multi-dimensional multi-class, respectively.
    In this case use the ``threshold`` argument to control the "binarization" of predictions.

    Args:
        num_classes:
            Number of classes.
        normalize: Normalization mode for confusion matrix. Choose from

        - ``None``: no normalization (default)
        - ``'true'``: normalization over the targets (most commonly used)
        - ``'pred'``: normalization over the predictions
        - ``'all'``: normalization over the whole matrix

        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5

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

        >>> from pytorch_lightning.metrics import ConfusionMatrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(num_classes=2)
        >>> confmat(preds, target)
        tensor([[2., 0.],
                [1., 1.]])

    """

    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
        threshold: float = 0.5,
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
        self.num_classes = num_classes
        self.normalize = normalize
        self.threshold = threshold

        allowed_normalize = ("true", "pred", "all", None)
        assert (
            self.normalize in allowed_normalize
        ), f"Argument average needs to one of the following: {allowed_normalize}"

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth values
        """
        confmat = _confusion_matrix_update(preds, target, num_classes=self.num_classes, threshold=self.threshold)
        self.confmat += confmat

    def compute(self) -> torch.Tensor:
        """
        Computes confusion matrix

        Return:
            Returns an `[C, C]` matrix, where `C` is the number of classes.
        """
        return _confusion_matrix_compute(self.confmat, self.normalize)
