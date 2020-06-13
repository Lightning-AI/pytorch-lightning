from typing import Any, Optional, Sequence, Tuple

import torch

from pytorch_lightning.metrics.functional.classification import (
    accuracy,
    confusion_matrix,
    precision_recall_curve,
    precision,
    recall,
    average_precision,
    auroc,
    fbeta_score,
    f1_score,
    roc,
    multiclass_roc,
    multiclass_precision_recall_curve,
    dice_score
)
from pytorch_lightning.metrics.metric import TensorMetric, TensorCollectionMetric

__all__ = [
    'Accuracy',
    'ConfusionMatrix',
    'PrecisionRecall',
    'Precision',
    'Recall',
    'AveragePrecision',
    'AUROC',
    'FBeta',
    'F1',
    'ROC',
    'MulticlassROC',
    'MulticlassPrecisionRecall',
    'DiceCoefficient'
]


class Accuracy(TensorMetric):
    """
    Computes the accuracy classification score

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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
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
            torch.Tensor: precision values
            torch.Tensor: recall values
            torch.Tensor: threshold values
        """
        return precision_recall_curve(pred=pred, target=target,
                                      sample_weight=sample_weight,
                                      pos_label=self.pos_label)


class Precision(TensorMetric):
    """
    Computes the precision score
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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
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
    """Computes the FBeta Score"""

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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction

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
    """Computes the F1 score"""

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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
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
            torch.Tensor: false positive rate
            torch.Tensor: true positive rate
            torch.Tensor: thresholds
        """
        return roc(pred=pred, target=target,
                   sample_weight=sample_weight,
                   pos_label=self.pos_label)


class MulticlassROC(TensorCollectionMetric):
    """
    Computes the multiclass ROC
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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
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
            pred: predicted labels
            target: groundtruth labels
            sample_weight: Weights for each sample defining the sample's impact on the score

        Return:
            tuple: A tuple consisting of one tuple per class,
                holding false positive rate, true positive rate and thresholds
        """
        return multiclass_roc(pred=pred,
                              target=target,
                              sample_weight=sample_weight,
                              num_classes=self.num_classes)


class MulticlassPrecisionRecall(TensorCollectionMetric):
    """Computes the multiclass PR Curve"""

    def __init__(
            self,
            num_classes: Optional[int] = None,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
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
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: Weights for each sample defining the sample's impact on the score

        Return:
            tuple: A tuple consisting of one tuple per class,
                holding precision, recall and thresholds
        """
        return multiclass_precision_recall_curve(pred=pred,
                                                 target=target,
                                                 sample_weight=sample_weight,
                                                 num_classes=self.num_classes)


class DiceCoefficient(TensorMetric):
    """Computes the dice coefficient"""

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
            reduction: a method for reducing accuracies over labels (default: takes the mean)
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
            pred: predicted labels
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
