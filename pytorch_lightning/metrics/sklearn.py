from typing import Any, Optional, Union, Sequence

import numpy as np
import torch

from pytorch_lightning import _logger as lightning_logger
from pytorch_lightning.metrics.metric import NumpyMetric


class SklearnMetric(NumpyMetric):
    def __init__(self, metric_name: str,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM, **kwargs):
        """
        Bridge between PyTorch Lightning and scikit-learn metrics

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        .. note::
            The order of targets and predictions may be different from the order typically used in PyTorch

        Args:
            metric_name: the metric name to import anc compute from scikit-learn.metrics
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
            **kwargs: additonal keyword arguments (will be forwarded to metric call)
        """
        super().__init__(name=metric_name, reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.metric_kwargs = kwargs

        lightning_logger.debug(
            'Every metric call will cause a GPU synchronization, which may slow down your code')

    @property
    def metric_fn(self):
        import sklearn.metrics
        return getattr(sklearn.metrics, self.name)

    def forward(self, *args, **kwargs) -> Union[np.ndarray, int, float]:
        """
        Carries the actual metric computation
        Args:
            *args: Positional arguments forwarded to metric call (should be already converted to numpy)
            **kwargs: keyword arguments forwarded to metric call (should be already converted to numpy)

        Returns:
            the metric value (will be converted to tensor by baseclass

        """
        return self.metric_fn(*args, **kwargs, **self.metric_kwargs)


class Accuracy(SklearnMetric):
    def __init__(self, normalize: bool = True,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Calculates the Accuracy Score

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        Args:
            normalize: If ``False``, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(metric_name='accuracy_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         normalize=normalize)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Computes the accuracy
        Args:
            y_pred: the array containing the predictions (already in categorical form)
            y_true: the array containing the targets (in categorical form)
            sample_weight:

        Returns:
            Accuracy Score


        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class AUC(SklearnMetric):
    def __init__(self, reorder: bool = False,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
        Calculates the Area Under the Curve using the trapoezoidal rule

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        Args:
            reorder: If ``True``, assume that the curve is ascending in the case of ties, as for an ROC curve.
                If the curve is non-ascending, the result will be wrong.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """

        super().__init__(metric_name='auc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         reorder=reorder)

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the AUC
        Args:
            x: x coordinates.
            y: y coordinates.

        Returns:
            AUC calculated with trapezoidal rule

        """
        return super().forward(x=x, y=y)


class AveragePrecision(SklearnMetric):
    def __init__(self, average: Optional[str] = 'macro',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
        Calculates the average precision (AP) score.
        Args:
            average: If None, the scores for each class are returned. Otherwise, this determines the type of
                averaging performed on the data:
                * If 'micro': Calculate metrics globally by considering each element of the label indicator
                    matrix as a label.
                * If 'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
                * If 'weighted': Calculate metrics for each label, and find their average, weighted by
                    support (the number of true instances for each label).
                * If 'samples': Calculate metrics for each instance, and find their average.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('average_precision_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         average=average)

    def forward(self, y_score: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> float:
        """

        Args:
            y_score: Target scores, can either be probability estimates of the positive class,
                confidence values, or binary decisions.
            y_true: True binary labels in binary label indicators.
            sample_weight: Sample weights.
        Returns:
            average precision score
        """
        return super().forward(y_score=y_score, y_true=y_true,
                               sample_weight=sample_weight)


class ConfusionMatric(SklearnMetric):
    def __init__(self, labels: Optional[Sequence] = None,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
        Compute confusion matrix to evaluate the accuracy of a classification
        By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
        is equal to the number of observations known to be in group :math:`i` but
        predicted to be in group :math:`j`.

        Args:
            labels: List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If none is given, those that appear at least once
                in ``y_true`` or ``y_pred`` are used in sorted order.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('confusion_matrix',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """

        Args:
            y_pred: Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.

        Returns: Confusion matrix (array of shape [n_classes, n_classes])

        """
        return super().forward(y_pred=y_pred, y_true=y_true)


class F1(SklearnMetric):
    """
    Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    References:
        .. [1] `Wikipedia entry for the F1-score
           <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(self, labels: Optional[Sequence] = None,
                 pos_labels: Union[str, int] = 1,
                 average: Optional[str] = 'binary',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """

        Args:
            labels: Integer array of labels.
            pos_labels: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:
                ``'binary'``:
                    Only report results for the class specified by ``pos_label``.
                    This is applicable only if targets (``y_{true,pred}``) are binary.
                ``'micro'``:
                    Calculate metrics globally by counting the total true positives,
                    false negatives and false positives.
                ``'macro'``:
                    Calculate metrics for each label, and find their unweighted
                    mean.  This does not take label imbalance into account.
                ``'weighted'``:
                    Calculate metrics for each label, and find their average, weighted
                    by support (the number of true instances for each label). This
                    alters 'macro' to account for label imbalance; it can result in an
                    F-score that is not between precision and recall.
                ``'samples'``:
                    Calculate metrics for each instance, and find their average (only
                    meaningful for multilabel classification where this differs from
                    :func:`accuracy_score`).
                Note that if ``pos_label`` is given in binary classification with
                `average != 'binary'`, only that positive class is reported. This
                behavior is deprecated and will change in version 0.18.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('f1_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels,
                         pos_labels=pos_labels,
                         average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """

        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.


        Returns: F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class FBeta(SklearnMetric):

    pass


class Precision(SklearnMetric):
    pass


class Recall(SklearnMetric):
    pass


class PrecisionRecallCurve(SklearnMetric):
    pass


class ROC(SklearnMetric):
    pass


class AUROC(SklearnMetric):
    pass


class R2(SklearnMetric):
    pass


class Jaccard(SklearnMetric):
    pass
