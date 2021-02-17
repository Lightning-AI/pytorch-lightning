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
from distutils.version import LooseVersion
from typing import Any, Callable, Optional

import torch

from pytorch_lightning.metrics.functional.auroc import _auroc_compute, _auroc_update
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn


class AUROC(Metric):
    r"""Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Further_interpretations>`_.
    Works for both binary, multilabel and multiclass problems. In the case of
    multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    For non-binary input, if the ``preds`` and ``target`` tensor have the same
    size the input will be interpretated as multilabel and if ``preds`` have one
    dimension more than the ``target`` tensor the input will be interpretated as
    multiclass.

     Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
        average:
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range [0, max_fpr]. Should be a float between 0 and 1.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Example (binary case):

        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> auroc = AUROC(pos_label=1)
        >>> auroc(preds, target)
        tensor(0.5000)

    Example (multiclass case):

        >>> preds = torch.tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = torch.tensor([0, 1, 1, 2, 2])
        >>> auroc = AUROC(num_classes=3)
        >>> auroc(preds, target)
        tensor(0.7778)

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[str] = 'macro',
        max_fpr: Optional[float] = None,
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
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, 'macro', 'weighted')
        if self.average not in allowed_average:
            raise ValueError(
                f'Argument `average` expected to be one of the following: {allowed_average} but got {average}'
            )

        if self.max_fpr is not None:
            if (not isinstance(max_fpr, float) and 0 < max_fpr <= 1):
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

            if LooseVersion(torch.__version__) < LooseVersion('1.6.0'):
                raise RuntimeError(
                    '`max_fpr` argument requires `torch.bucketize` which is not available below PyTorch version 1.6'
                )

        self.mode = None
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            'Metric `AUROC` will save all targets and predictions in buffer.'
            ' For large datasets this may lead to large memory footprint.'
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        preds, target, mode = _auroc_update(preds, target)

        self.preds.append(preds)
        self.target.append(target)

        if self.mode is not None and self.mode != mode:
            raise ValueError(
                'The mode of data (binary, multi-label, multi-class) should be constant, but changed'
                f' between batches from {self.mode} to {mode}'
            )
        self.mode = mode

    def compute(self) -> torch.Tensor:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        return _auroc_compute(
            preds,
            target,
            self.mode,
            self.num_classes,
            self.pos_label,
            self.average,
            self.max_fpr,
        )
