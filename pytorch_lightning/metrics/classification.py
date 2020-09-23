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
    _confmat_normalize,
    dice_score,
    f1_score,
    fbeta_score,
    iou,
    multiclass_precision_recall_curve,
    multiclass_roc,
    precision_recall_curve,
    roc,
    precision_recall
)
from pytorch_lightning.metrics.functional.reduction import class_reduce
from pytorch_lightning.metrics.metric import TensorMetric


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
        class_reduction: str = 'micro',
        reduce_group: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            class_reduction: method to reduce metric score over labels

                - ``'micro'``: calculate metrics globally (default)
                - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
                - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
                - ``'none'``: returns calculated metric per class

            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(name="accuracy", reduce_group=reduce_group)
        self.num_classes = num_classes
        assert class_reduction in ('micro', 'macro', 'weighted', 'none')
        self.class_reduction = class_reduction

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
                        num_classes=self.num_classes,
                        class_reduction='none',
                        return_state=True)

    @staticmethod
    def compute(self, data: Any, output: Any):
        tps, sups = output['tps'], output['sups']
        return class_reduce(tps, sups, sups, class_reduction=self.class_reduction)


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
        num_classes: Optional[int] = None,
        normalize: bool = False,
        reduce_group: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            normalize: whether to compute a normalized confusion matrix
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="confusion_matrix",
            reduce_group=reduce_group,
        )
        self.normalize = normalize
        self.num_classes = num_classes

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
                                normalize=False,  # we normalize after ddp sync
                                num_classes=self.num_classes)

    @staticmethod
    def compute(self, data: Any, output: Any):
        """ Confusion matrix normalization needs to happen after ddp sync """
        confmat = output
        if self.normalize:
            confmat = _confmat_normalize(confmat)
        return confmat


class PrecisionRecallCurve(TensorMetric):
    """
    Computes the precision recall curve

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = PrecisionRecallCurve()
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
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="precision_recall_curve",
            reduce_group=reduce_group,
        )

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
                                      sample_weight=sample_weight, pos_label=self.pos_label)


class Precision(TensorMetric):
    """
    Computes the precision score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Precision(num_classes=4, class_reduction='macro')
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
        reduce_group: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            class_reduction: method to reduce metric score over labels

                - ``'micro'``: calculate metrics globally (default)
                - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
                - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
                - ``'none'``: returns calculated metric per class

            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="precision",
            reduce_group=reduce_group,
        )
        self.num_classes = num_classes
        assert class_reduction in ('micro', 'macro', 'weighted', 'none')
        self.class_reduction = class_reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return precision_recall(pred=pred, target=target,
                                num_classes=self.num_classes,
                                class_reduction='none',
                                return_state=True)

    @staticmethod
    def compute(self, data: Any, output: Any):
        tps, fps, sups = output['tps'], output['fps'], output['sups']
        return class_reduce(tps, tps + fps, sups, class_reduction=self.class_reduction)


class Recall(TensorMetric):
    """
    Computes the recall score

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Recall()
        >>> metric(pred, target)
        tensor(0.7500)

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
        reduce_group: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            class_reduction: method to reduce metric score over labels

                - ``'micro'``: calculate metrics globally (default)
                - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
                - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
                - ``'none'``: returns calculated metric per class

            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="recall",
            reduce_group=reduce_group,
        )

        self.num_classes = num_classes
        assert class_reduction in ('micro', 'macro', 'weighted', 'none')
        self.class_reduction = class_reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels

        Return:
            A Tensor with the classification score.
        """
        return precision_recall(pred=pred, target=target,
                                num_classes=self.num_classes,
                                class_reduction='none',
                                return_state=True)

    @staticmethod
    def compute(self, data: Any, output: Any):
        tps, fns, sups = output['tps'], output['fns'], output['sups']
        return class_reduce(tps, tps + fns, sups, class_reduction=self.class_reduction)


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
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="AP",
            reduce_group=reduce_group,
        )

        self.pos_label = pos_label

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, sample_weight: Optional[Sequence] = None
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
        return average_precision(pred=pred, target=target, sample_weight=sample_weight, pos_label=self.pos_label)


class AUROC(TensorMetric):
    """
    Computes the area under curve (AUC) of the receiver operator characteristic (ROC)

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = AUROC()
        >>> metric(pred, target)
        tensor(0.5000)

    """

    def __init__(
        self,
        pos_label: int = 1,
        reduce_group: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="auroc",
            reduce_group=reduce_group,
        )

        self.pos_label = pos_label

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, sample_weight: Optional[Sequence] = None
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
        return auroc(pred=pred, target=target, sample_weight=sample_weight, pos_label=self.pos_label)


class FBeta(TensorMetric):
    """
    Computes the FBeta Score, which is the weighted harmonic mean of precision and recall.
        It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = FBeta(0.25, class_reduction='macro')
        >>> metric(pred, target)
        tensor(0.7361)
    """

    def __init__(
        self,
        beta: float,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
        reduce_group: Any = None,
    ):
        """
        Args:
            beta: determines the weight of recall in the combined score.
            num_classes: number of classes
            class_reduction: method to reduce metric score over labels

                - ``'micro'``: calculate metrics globally (default)
                - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
                - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
                - ``'none'``: returns calculated metric per class

            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="fbeta",
            reduce_group=reduce_group,
        )

        self.beta = beta
        self.num_classes = num_classes
        assert class_reduction in ('micro', 'macro', 'weighted', 'none')
        self.class_reduction = class_reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: groundtruth labels

        Return:
            torch.Tensor: classification score
        """
        return precision_recall(pred=pred, target=target,
                                num_classes=self.num_classes,
                                class_reduction='none',
                                return_state=True)

    @staticmethod
    def compute(self, data: Any, output: Any):
        """ tps, fps, fns, sups needs to be synced before we do any calculations """
        tps, fps, fns, sups = output['tps'], output['fps'], output['fns'], output['sups']

        intermidiate_reduction = 'none' if self.class_reduction != "micro" else 'micro'
        precision = class_reduce(tps, tps + fps, sups, class_reduction=intermidiate_reduction)
        recall = class_reduce(tps, tps + fns, sups, class_reduction=intermidiate_reduction)

        num = (1 + self.beta ** 2) * precision * recall
        denom = ((self.beta ** 2) * precision + recall)
        if intermidiate_reduction == 'micro':
            return torch.sum(num) / torch.sum(denom)
        return class_reduce(num, denom, sups, class_reduction=self.class_reduction)


class F1(FBeta):
    """
    Computes the F1 score, which is the harmonic mean of the precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = F1(class_reduction='macro')
        >>> metric(pred, target)
        tensor(0.6667)
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_reduction: str = 'micro',
        reduce_group: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            class_reduction: method to reduce metric score over labels

                - ``'micro'``: calculate metrics globally (default)
                - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
                - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
                - ``'none'``: returns calculated metric per class

            reduce_group: the process group to reduce metric results from DDP
        """
        super(TensorMetric, self).__init__(
            name="f1",
            reduce_group=reduce_group,
        )

        self.num_classes = num_classes
        assert class_reduction in ('micro', 'macro', 'weighted', 'none')
        self.class_reduction = class_reduction
        self.beta = 1.0


class ROC(TensorMetric):
    """
    Computes the Receiver Operator Characteristic (ROC)

    Example:

        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = ROC()
        >>> metric(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([4., 3., 2., 1., 0.]))

    """

    def __init__(
        self,
        pos_label: int = 1,
        reduce_group: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="roc",
            reduce_group=reduce_group,
        )

        self.pos_label = pos_label

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, sample_weight: Optional[Sequence] = None
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
        return roc(pred=pred, target=target, sample_weight=sample_weight, pos_label=self.pos_label)


class MulticlassROC(TensorMetric):
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
    ):
        """
        Args:
            num_classes: number of classes
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="multiclass_roc",
            reduce_group=reduce_group,
        )

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
            pred: predicted probability for each label
            target: groundtruth labels
            sample_weight: Weights for each sample defining the sample's impact on the score

        Return:
            tuple: A tuple consisting of one tuple per class, holding false positive rate, true positive rate and thresholds

        """
        return multiclass_roc(pred=pred, target=target, sample_weight=sample_weight, num_classes=self.num_classes)

    def aggregate(self, *tensors: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Aggregates results by stacking them instead of concatenating before averaging.

        Returns:
            the aggregated results
        """

        return tuple([tuple([torch.stack(tmps).mean(0) for tmps in zip(*_tensors)]) for _tensors in zip(*tensors)])


class MulticlassPrecisionRecallCurve(TensorMetric):
    """Computes the multiclass PR Curve

    Example:

        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                     [0.05, 0.85, 0.05, 0.05],
        ...                     [0.05, 0.05, 0.85, 0.05],
        ...                     [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> metric = MulticlassPrecisionRecallCurve()
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
    ):
        """
        Args:
            num_classes: number of classes
            reduce_group: the process group to reduce metric results from DDP

        """
        super().__init__(
            name="multiclass_precision_recall_curve",
            reduce_group=reduce_group,
        )

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
        return multiclass_precision_recall_curve(
            pred=pred, target=target, sample_weight=sample_weight, num_classes=self.num_classes
        )

    def aggregate(self, *tensors: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Aggregates results by stacking them instead of concatenating before averaging.

        Returns:
            the aggregated results
        """

        return tuple([tuple([torch.stack(tmps).mean(0) for tmps in zip(*_tensors)]) for _tensors in zip(*tensors)])


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
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
        reduce_group: Any = None,
    ):
        """
        Args:
            include_background: whether to also compute dice for the background
            nan_score: score to return, if a NaN occurs during computation (denom zero)
            no_fg_score: score to return, if no foreground pixel was found in target
            reduction: a method to reduce metric score over labels.

                - ``'elementwise_mean'``: takes the mean (default)
                - ``'sum'``: takes the sum
                - ``'none'``: no reduction will be applied
            reduce_group: the process group to reduce metric results from DDP
        """
        super().__init__(
            name="dice",
            reduce_group=reduce_group,
        )

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
        return dice_score(
            pred=pred,
            target=target,
            bg=self.include_background,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
        )


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
            ignore_index: Optional[int] = None,
            absent_score: float = 0.0,
            num_classes: Optional[int] = None,
            reduction: str = "elementwise_mean",
    ):
        """
        Args:
            ignore_index: optional int specifying a target class to ignore. If given, this class index does not
                contribute to the returned score, regardless of reduction method. Has no effect if given an int that is
                not in the range [0, num_classes-1], where num_classes is either given or derived from pred and target.
                By default, no index is ignored, and all classes are used.
            absent_score: score to use for an individual class, if no instances of the class index were present in
                `y_pred` AND no instances of the class index were present in `y_true`. For example, if we have 3
                classes, [0, 0] for `y_pred`, and [0, 2] for `y_true`, then class 1 would be assigned the
                `absent_score`. Default is 0.0.
            num_classes: Optionally specify the number of classes
            reduction: a method to reduce metric score over labels.

                - ``'elementwise_mean'``: takes the mean (default)
                - ``'sum'``: takes the sum
                - ``'none'``: no reduction will be applied
        """
        super().__init__(name="iou")
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        """
        Actual metric calculation.
        """
        return iou(
            pred=y_pred,
            target=y_true,
            ignore_index=self.ignore_index,
            absent_score=self.absent_score,
            num_classes=self.num_classes,
            reduction=self.reduction,
        )
