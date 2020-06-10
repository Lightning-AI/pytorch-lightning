from typing import Any, Optional, Union, Sequence

import numpy as np
import torch

from pytorch_lightning import _logger as lightning_logger
from pytorch_lightning.metrics.metric import NumpyMetric

__all__ = [
    'SklearnMetric',
    'Accuracy',
    'AveragePrecision',
    'AUC',
    'ConfusionMatrix',
    'F1',
    'FBeta',
    'Precision',
    'Recall',
    'PrecisionRecallCurve',
    'ROC',
    'AUROC'
]


class SklearnMetric(NumpyMetric):
    """
    Bridge between PyTorch Lightning and scikit-learn metrics

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code

    Note:
        The order of targets and predictions may be different from the order typically used in PyTorch
    """
    def __init__(self, metric_name: str,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM, **kwargs):
        """
        Args:
            metric_name: the metric name to import and compute from scikit-learn.metrics
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
            f'Metric {self.__class__.__name__} is using Sklearn as backend, meaning that'
            ' every metric call will cause a GPU synchronization, which may slow down your code'
        )

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

        Return:
            the metric value (will be converted to tensor by baseclass)

        """
        return self.metric_fn(*args, **kwargs, **self.metric_kwargs)


class Accuracy(SklearnMetric):
    """
    Calculates the Accuracy Score

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code
    """
    def __init__(self, normalize: bool = True,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
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
            sample_weight:  Sample weights.

        Return:
            Accuracy Score

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class AUC(SklearnMetric):
    """
    Calculates the Area Under the Curve using the trapoezoidal rule

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code
    """
    def __init__(self,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """

        super().__init__(metric_name='auc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         )

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the AUC

        Args:
            x: x coordinates.
            y: y coordinates.

        Return:
            AUC calculated with trapezoidal rule

        """
        return super().forward(x=x, y=y)


class AveragePrecision(SklearnMetric):
    """
    Calculates the average precision (AP) score.
    """
    def __init__(self, average: Optional[str] = 'macro',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
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

        Return:
            average precision score
        """
        return super().forward(y_score=y_score, y_true=y_true,
                               sample_weight=sample_weight)


class ConfusionMatrix(SklearnMetric):
    """
    Compute confusion matrix to evaluate the accuracy of a classification
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.
    """
    def __init__(self, labels: Optional[Sequence] = None,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
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

        Return:
            Confusion matrix (array of shape [n_classes, n_classes])

        """
        return super().forward(y_pred=y_pred, y_true=y_true)


class F1(SklearnMetric):
    r"""
    Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is:

    .. math::

        F_1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}

    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    References
        - [1] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(self, labels: Optional[Sequence] = None,
                 pos_label: Union[str, int] = 1,
                 average: Optional[str] = 'binary',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
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
                         pos_label=pos_label,
                         average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class FBeta(SklearnMetric):
    """
    Compute the F-beta score. The `beta` parameter determines the weight of precision in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> inf``
    only recall).

    References:
        - [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
          Modern Information Retrieval. Addison Wesley, pp. 327-328.
        - [2] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(self, beta: float, labels: Optional[Sequence] = None,
                 pos_label: Union[str, int] = 1,
                 average: Optional[str] = 'binary',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            beta: Weight of precision in harmonic mean.
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
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
        super().__init__('fbeta_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         beta=beta,
                         labels=labels,
                         pos_label=pos_label,
                         average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.


        Return:
            FBeta score of the positive class in binary classification or weighted
            average of the FBeta scores of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class Precision(SklearnMetric):
    """
    Compute the precision
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The best value is 1 and the worst value is 0.
    """

    def __init__(self, labels: Optional[Sequence] = None,
                 pos_label: Union[str, int] = 1,
                 average: Optional[str] = 'binary',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
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
        super().__init__('precision_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels,
                         pos_label=pos_label,
                         average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Precision of the positive class in binary classification or weighted
            average of the precision of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class Recall(SklearnMetric):
    """
    Compute the recall
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    """

    def __init__(self, labels: Optional[Sequence] = None,
                 pos_label: Union[str, int] = 1,
                 average: Optional[str] = 'binary',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            labels: Integer array of labels.
            pos_label: The class to report if ``average='binary'``.
            average: This parameter is required for multiclass/multilabel targets.
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                * ``'binary'``:
                  Only report results for the class specified by ``pos_label``.
                  This is applicable only if targets (``y_{true,pred}``) are binary.
                * ``'micro'``:
                  Calculate metrics globally by counting the total true positives,
                  false negatives and false positives.
                * ``'macro'``:
                  Calculate metrics for each label, and find their unweighted
                  mean.  This does not take label imbalance into account.
                * ``'weighted'``:
                  Calculate metrics for each label, and find their average, weighted
                  by support (the number of true instances for each label). This
                  alters 'macro' to account for label imbalance; it can result in an
                  F-score that is not between precision and recall.
                * ``'samples'``:
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
        super().__init__('recall_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels,
                         pos_label=pos_label,
                         average=average)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Recall of the positive class in binary classification or weighted
            average of the recall of each class for the multiclass task.

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class PrecisionRecallCurve(SklearnMetric):
    """
    Compute precision-recall pairs for different probability thresholds

    Note:
        This implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    x axis.
    """

    def __init__(self,
                 pos_label: Union[str, int] = 1,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            pos_label: The class to report if ``average='binary'``.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('precision_recall_curve',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         pos_label=pos_label)

    def forward(self, probas_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            probas_pred : Estimated probabilities or decision function.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Returns:
            precision:
                Precision values such that element i is the precision of
                predictions with score >= thresholds[i] and the last element is 1.
            recall:
                Decreasing recall values such that element i is the recall of
                predictions with score >= thresholds[i] and the last element is 0.
            thresholds:
                Increasing thresholds on the decision function used to compute
                precision and recall.

        """
        # only return x and y here, since for now we cannot auto-convert elements of multiple length.
        # Will be fixed in native implementation
        return np.array(
            super().forward(probas_pred=probas_pred, y_true=y_true, sample_weight=sample_weight)[:2])


class ROC(SklearnMetric):
    """
    Compute Receiver operating characteristic (ROC)

    Note:
        this implementation is restricted to the binary classification task.
    """

    def __init__(self,
                 pos_label: Union[str, int] = 1,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Args:
            pos_labels: The class to report if ``average='binary'``.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.

        References:
            - [1] `Wikipedia entry for the Receiver operating characteristic
              <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
        """
        super().__init__('roc_curve',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         pos_label=pos_label)

    def forward(self, y_score: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Args:
            y_score : Target scores, can either be probability estimates of the positive
                class or confidence values.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Returns:
            fpr:
                Increasing false positive rates such that element i is the false
                positive rate of predictions with score >= thresholds[i].
            tpr:
                Increasing true positive rates such that element i is the true
                positive rate of predictions with score >= thresholds[i].
            thresholds:
                Decreasing thresholds on the decision function used to compute
                fpr and tpr. `thresholds[0]` represents no instances being predicted
                and is arbitrarily set to `max(y_score) + 1`.

        """
        return np.array(super().forward(y_score=y_score, y_true=y_true, sample_weight=sample_weight)[:2])


class AUROC(SklearnMetric):
    """
    Compute Area Under the Curve (AUC) from prediction scores

    Note:
        this implementation is restricted to the binary classification task
        or multilabel classification task in label indicator format.
    """

    def __init__(self, average: Optional[str] = 'macro',
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
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
        super().__init__('roc_auc_score',
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

        Return:
            Area Under Receiver Operating Characteristic Curve
        """
        return super().forward(y_score=y_score, y_true=y_true,
                               sample_weight=sample_weight)
