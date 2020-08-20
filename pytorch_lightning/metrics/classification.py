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

from typing import Any, Optional, Sequence, Tuple

import torch

from pytorch_lightning.metrics.functional.classification import (
    accuracy,
    auroc,
    average_precision,
    confusion_matrix,
    dice_score,
    f1_score,
    fbeta_score,
    iou,
    multiclass_precision_recall_curve,
    multiclass_roc,
    precision,
    precision_recall_curve,
    recall,
    roc
)
from pytorch_lightning.metrics.metric import TensorCollectionMetric, TensorMetric


class Accuracy(TensorMetric):
    """
    Computes the accuracy classification score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Accuracy()
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='accuracy',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return accuracy(pred=pred, target=target,
                        num_classes=self.num_classes, reduction=self.reduction)


class ConfusionMatrix(TensorMetric):
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 2])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = ConfusionMatrix()
        >>> metric(pred, target)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 2.]])

    """

    def __init__(
            self,
            normalize: bool = False,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            normalize: whether to compute a normalized confusion matrix
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='confusion_matrix',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the confusion matrix.
        """
        return confusion_matrix(pred=pred, target=target,
                                normalize=self.normalize)


class PrecisionRecall(TensorCollectionMetric):
    """
    Computes the precision recall curve

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = PrecisionRecall()
        >>> prec, recall, thr = metric(pred, target)
        >>> prec
        tensor([0.3333, 0.0000, 0.0000, 1.0000])
        >>> recall
        tensor([1., 0., 0., 0.])
        >>> thr
        tensor([1., 2., 3.])

    """

    def __init__(
            self,
            pos_label: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='precision_recall_curve',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_label = pos_label

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample

        Return:
            - precision values
            - recall values
            - threshold values
        """
        return precision_recall_curve(pred=pred, target=target,
                                      sample_weight=sample_weight,
                                      pos_label=self.pos_label)


class Precision(TensorMetric):
    """
    Computes the precision score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Precision(num_classes=4)
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='precision',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return precision(pred=pred, target=target,
                         num_classes=self.num_classes,
                         reduction=self.reduction)


class Recall(TensorMetric):
    """
    Computes the recall score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Recall()
        >>> metric(pred, target)
        tensor(0.6250)

    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='recall',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return recall(pred=pred,
                      target=target,
                      num_classes=self.num_classes,
                      reduction=self.reduction)


class AveragePrecision(TensorMetric):
    """
    Computes the average precision score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = AveragePrecision()
        >>> metric(pred, target)
        tensor(0.3333)

    """

    def __init__(
            self,
            pos_label: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='AP',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_label = pos_label

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None
    ) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample

        Return:
            torch.Tensor: classification score
        """
        return average_precision(pred=pred, target=target,
                                 sample_weight=sample_weight,
                                 pos_label=self.pos_label)


class AUROC(TensorMetric):
    """
    Computes the area under curve (AUC) of the receiver operator characteristic (ROC)

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = AUROC()
        >>> metric(pred, target)
        tensor(0.3333)

    """

    def __init__(
            self,
            pos_label: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='auroc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_label = pos_label

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None
    ) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample

        Return:
            torch.Tensor: classification score
        """
        return auroc(pred=pred, target=target,
                     sample_weight=sample_weight,
                     pos_label=self.pos_label)


class FBeta(TensorMetric):
    """
    Computes the FBeta Score, which is the weighted harmonic mean of precision and recall.
        It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = FBeta(0.25)
        >>> metric(pred, target)
        tensor(0.7361)
    """

    def __init__(
            self,
            beta: float,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            beta: determines the weight of recall in the combined score.
            num_classes: number of classes
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for DDP reduction
        """
        super().__init__(name='fbeta',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels

        Return:
            torch.Tensor: classification score
        """
        return fbeta_score(pred=pred, target=target,
                           beta=self.beta, num_classes=self.num_classes,
                           reduction=self.reduction)


class F1(TensorMetric):
    """
    Computes the F1 score, which is the harmonic mean of the precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = F1()
        >>> metric(pred, target)
        tensor(0.6667)
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='f1',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels

        Return:
            torch.Tensor: classification score
        """
        return f1_score(pred=pred, target=target,
                        num_classes=self.num_classes,
                        reduction=self.reduction)


class ROC(TensorCollectionMetric):
    """
    Computes the Receiver Operator Characteristic (ROC)

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = ROC()
        >>> fps, tps, thresholds = metric(pred, target)
        >>> fps
        tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000])
        >>> tps
        tensor([0., 0., 0., 1., 1.])
        >>> thresholds
        tensor([4., 3., 2., 1., 0.])

    """

    def __init__(
            self,
            pos_label: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='roc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_label = pos_label

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample

        Return:
            - false positive rate
            - true positive rate
            - thresholds
        """
        return roc(pred=pred, target=target,
                   sample_weight=sample_weight,
                   pos_label=self.pos_label)


class MulticlassROC(TensorCollectionMetric):
    """
    Computes the multiclass ROC

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                     [0.05, 0.85, 0.05, 0.05],
        ...                     [0.05, 0.05, 0.85, 0.05],
        ...                     [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassROC()
        >>> classes_roc = metric(pred, target)
        >>> metric(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        ((tensor([0., 0., 1.]), tensor([0., 1., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0., 0., 1.]), tensor([0., 1., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0.0000, 0.3333, 1.0000]), tensor([0., 0., 1.]), tensor([1.8500, 0.8500, 0.0500])),
         (tensor([0.0000, 0.3333, 1.0000]), tensor([0., 0., 1.]), tensor([1.8500, 0.8500, 0.0500])))
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='multiclass_roc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.num_classes = num_classes

    def forward(
            self, pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Actual metric computation

        Args:
            pred: predicted probability for each label
            target: groundtruth labels
            sample_weight: Weights for each sample defining the sample's impact on the score

        Return:
            tuple: A tuple consisting of one tuple per class, holding false positive rate, true positive rate and thresholds

        """
        return multiclass_roc(pred=pred,
                              target=target,
                              sample_weight=sample_weight,
                              num_classes=self.num_classes)


class MulticlassPrecisionRecall(TensorCollectionMetric):
    """Computes the multiclass PR Curve

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                     [0.05, 0.85, 0.05, 0.05],
        ...                     [0.05, 0.05, 0.85, 0.05],
        ...                     [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassPrecisionRecall()
        >>> metric(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        ((tensor([1., 1.]), tensor([1., 0.]), tensor([0.8500])),
         (tensor([1., 1.]), tensor([1., 0.]), tensor([0.8500])),
         (tensor([0.2500, 0.0000, 1.0000]), tensor([1., 0., 0.]), tensor([0.0500, 0.8500])),
         (tensor([0.2500, 0.0000, 1.0000]), tensor([1., 0., 0.]), tensor([0.0500, 0.8500])))
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction

        """
        super().__init__(name='multiclass_precision_recall_curve',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.num_classes = num_classes

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actual metric computation

        Args:
            pred: predicted probability for each label
            target: groundtruth labels
            sample_weight: Weights for each sample defining the sample's impact on the score

        Return:
            tuple: A tuple consisting of one tuple per class, holding precision, recall and thresholds

        """
        return multiclass_precision_recall_curve(pred=pred,
                                                 target=target,
                                                 sample_weight=sample_weight,
                                                 num_classes=self.num_classes)


class DiceCoefficient(TensorMetric):
    """
    Computes the dice coefficient

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = DiceCoefficient()
        >>> metric(pred, target)
        tensor(0.3333)
    """

    def __init__(
            self,
            include_background: bool = False,
            nan_score: float = 0.0, no_fg_score: float = 0.0,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            include_background: whether to also compute dice for the background
            nan_score: score to return, if a NaN occurs during computation (denom zero)
            no_fg_score: score to return, if no foreground pixel was found in target
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='dice',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted probability for each label
            target: groundtruth labels

        Return:
            torch.Tensor: the calculated dice coefficient
        """
        return dice_score(pred=pred,
                          target=target,
                          bg=self.include_background,
                          nan_score=self.nan_score,
                          no_fg_score=self.no_fg_score,
                          reduction=self.reduction)


class IoU(TensorMetric):
    """
    Computes the intersection over union.

    Example:

        >>> pred = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                      [0, 0, 1, 1, 1, 0, 0, 0],
        ...                      [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> target = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
        ...                        [0, 0, 0, 1, 1, 1, 0, 0],
        ...                        [0, 0, 0, 0, 0, 0, 0, 0]])
        >>> metric = IoU()
        >>> metric(pred, target)
        tensor(0.7045)

    """

    def __init__(
            self,
            remove_bg: bool = False,
            reduction: str = 'elementwise_mean'
    ):
        """
        Args:
            remove_bg: Flag to state whether a background class has been included
                within input parameters. If true, will remove background class. If
                false, return IoU over all classes.
                Assumes that background is '0' class in input tensor
            reduction: a method to reduce metric score over labels (default: takes the mean)
                Available reduction methods:

                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='iou')
        self.remove_bg = remove_bg
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None):
        """
        Actual metric calculation.
        """
        return iou(y_pred, y_true, remove_bg=self.remove_bg, reduction=self.reduction)
