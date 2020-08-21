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

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from pytorch_lightning import _logger as lightning_logger
from pytorch_lightning.metrics.metric import NumpyMetric
from pytorch_lightning.utilities import rank_zero_warn

try:
    from torch.distributed import ReduceOp, group
except ImportError:
    class ReduceOp:
        SUM = None

    class group:
        WORLD = None

    rank_zero_warn('Unsupported `ReduceOp` for distributed computing.')


class SklearnMetric(NumpyMetric):
    """
    Bridge between PyTorch Lightning and scikit-learn metrics

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code

    Note:
        The order of targets and predictions may be different from the order typically used in PyTorch
    """

    def __init__(
            self,
            metric_name: str,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
            **kwargs,
    ):
        """
        Args:
            metric_name: the metric name to import and compute from scikit-learn.metrics
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
            **kwargs: additonal keyword arguments (will be forwarded to metric call)
        """
        super().__init__(name=metric_name,
                         reduce_group=reduce_group,
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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = Accuracy()
        >>> metric(y_pred, y_true)
        tensor([0.7500])

    """

    def __init__(
            self,
            normalize: bool = True,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = AUC()
        >>> metric(y_pred, y_true)
        tensor([4.])
    """

    def __init__(
            self,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
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
                         reduce_op=reduce_op)

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

    def __init__(
            self,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
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

    def forward(
            self,
            y_score: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
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


class BalancedAccuracy(SklearnMetric):
    """ Compute the balanced accuracy score

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([0, 0, 0, 1])
        >>> y_true = torch.tensor([0, 0, 1, 1])
        >>> metric = BalancedAccuracy()
        >>> metric(y_pred, y_true)
        tensor([0.7500])

    """

    def __init__(
            self,
            adjusted: bool = False,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            adjusted: If ``True``, the result sis adjusted for chance, such that random performance
                corresponds to 0 and perfect performance corresponds to 1
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('balanced_accuracy_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         adjusted=adjusted)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Args:
            y_pred: the array containing the predictions (already in categorical form)
            y_true: the array containing the targets (in categorical form)
            sample_weight:  Sample weights.

        Return:
            balanced accuracy score

        """
        return super().forward(y_true=y_true,
                               y_pred=y_pred,
                               sample_weight=sample_weight)


class CohenKappaScore(SklearnMetric):
    """
    Calculates Cohens kappa: a statitic that measures inter-annotator agreement

    Example:

        >>> y_pred = torch.tensor([1, 2, 0, 2])
        >>> y_true = torch.tensor([2, 2, 2, 1])
        >>> metric = CohenKappaScore()
        >>> metric(y_pred, y_true)
        tensor([-0.3333])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            weights: Optional[str] = None,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            labels: List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If none is given, those that appear at least once
                in ``y1`` or ``y2`` are used in sorted order.
            weights: string indicating weightning type used in scoring. None
                means no weighting, string ``linear`` means linear weighted
                and ``quadratic`` means quadratic weighted
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('cohen_kappa_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels,
                         weights=weights)

    def forward(
            self,
            y1: np.ndarray,
            y2: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Args:
            y_1: Labels assigned by first annotator
            y_2: Labels assigned by second annotator
            sample_weight:  Sample weights.

        Return:
            Cohens kappa score
        """
        return super().forward(y1=y1, y2=y2, sample_weight=sample_weight)


class ConfusionMatrix(SklearnMetric):
    """
    Compute confusion matrix to evaluate the accuracy of a classification
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 1])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = ConfusionMatrix()
        >>> metric(y_pred, y_true)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 1.]])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
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
            Confusion matrix (array of shape [num_classes, num_classes])

        """
        return super().forward(y_pred=y_pred, y_true=y_true)


class DCG(SklearnMetric):
    """ Compute discounted cumulative gain

    Warning:
        Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_score = torch.tensor([[.1, .2, .3, 4, 70]])
        >>> y_true = torch.tensor([[10, 0, 0, 1, 5]])
        >>> metric = DCG()
        >>> metric(y_score, y_true)
        tensor([9.4995])
    """

    def __init__(
            self,
            k: Optional[int] = None,
            log_base: float = 2,
            ignore_ties: bool = False,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            k: only consider the hightest k score in the ranking
            log_base: base of the logarithm used for the discount
            ignore_ties: If ``True``, assume there are no ties in y_score for efficiency gains
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('dcg_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         k=k,
                         log_base=log_base,
                         ignore_ties=ignore_ties)

    def forward(
            self,
            y_score: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Args:
            y_score: target scores, either probability estimates, confidence values
                or or non-thresholded measure of decisions
            y_true: Ground truth (correct) target values.
            sample_weight:  Sample weights.

        Return:
            DCG score

        """
        return super().forward(y_true=y_true,
                               y_score=y_score,
                               sample_weight=sample_weight)


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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = F1()
        >>> metric(y_pred, y_true)
        tensor([0.6667])

    References
        - [1] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            pos_label: Union[str, int] = 1,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = FBeta(beta=0.25)
        >>> metric(y_pred, y_true)
        tensor([0.7361])

    References:
        - [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
          Modern Information Retrieval. Addison Wesley, pp. 327-328.
        - [2] `Wikipedia entry for the F1-score
          <http://en.wikipedia.org/wiki/F1_score>`_
    """

    def __init__(
            self,
            beta: float,
            labels: Optional[Sequence] = None,
            pos_label: Union[str, int] = 1,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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


class Hamming(SklearnMetric):
    """
    Computes the average hamming loss

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([1, 1, 2, 3])
        >>> metric = Hamming()
        >>> metric(y_pred, y_true)
        tensor([0.2500])

    """

    def __init__(
            self,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.

        """
        super().__init__('hamming_loss',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Average hamming loss

        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)


class Hinge(SklearnMetric):
    """
    Computes the average hinge loss

    Example:

        >>> pred_decision = torch.tensor([-2.17, -0.97, -0.19, -0.43])
        >>> y_true = torch.tensor([1, 1, 0, 0])
        >>> metric = Hinge()
        >>> metric(pred_decision, y_true)
        tensor([1.6300])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            labels: Integer array of labels.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('hinge_loss',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels)

    def forward(
            self,
            pred_decision: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Args:
            pred_decision : Predicted decisions
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Average hinge loss

        """
        return super().forward(pred_decision=pred_decision,
                               y_true=y_true,
                               sample_weight=sample_weight)


class Jaccard(SklearnMetric):
    """
    Calculates jaccard similarity coefficient score

    Example:

        >>> y_pred = torch.tensor([1, 1, 1])
        >>> y_true = torch.tensor([0, 1, 1])
        >>> metric = Jaccard()
        >>> metric(y_pred, y_true)
        tensor([0.3333])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            pos_label: Union[str, int] = 1,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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
        super().__init__('jaccard_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         labels=labels,
                         pos_label=pos_label,
                         average=average)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
        """
        Args:
            y_pred : Estimated targets as returned by a classifier.
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Jaccard similarity score

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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = Precision()
        >>> metric(y_pred, y_true)
        tensor([0.7500])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            pos_label: Union[str, int] = 1,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = Recall()
        >>> metric(y_pred, y_true)
        tensor([0.6250])

    """

    def __init__(
            self,
            labels: Optional[Sequence] = None,
            pos_label: Union[str, int] = 1,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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

    def __init__(
            self,
            pos_label: Union[str, int] = 1,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
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

    def forward(
            self,
            probas_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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
        return np.array(super().forward(probas_pred=probas_pred,
                                        y_true=y_true,
                                        sample_weight=sample_weight)[:2])


class ROC(SklearnMetric):
    """
    Compute Receiver operating characteristic (ROC)

    Note:
        this implementation is restricted to the binary classification task.

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([0, 1, 2, 3])
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> metric = ROC()
        >>> fps, tps = metric(y_pred, y_true)
        >>> fps
        tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000])
        >>> tps
        tensor([0., 0., 0., 1., 1.])

    References:
        - [1] `Wikipedia entry for the Receiver operating characteristic
          <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    """

    def __init__(
            self,
            pos_label: Union[str, int] = 1,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            pos_labels: The class to report if ``average='binary'``.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('roc_curve',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         pos_label=pos_label)

    def forward(
            self,
            y_score: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
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

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    """

    def __init__(
            self,
            average: Optional[str] = 'macro',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
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

    def forward(
            self,
            y_score: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ) -> float:
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


class ExplainedVariance(SklearnMetric):
    """
    Calculates explained variance score

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 0.0, 2, 8])
        >>> y_true = torch.tensor([3, -0.5, 2, 7])
        >>> metric = ExplainedVariance()
        >>> metric(y_pred, y_true)
        tensor([0.9572])
    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'variance_weighted',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’, 'variance_weighted']
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('explained_variance_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Explained variance score

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)


class MeanAbsoluteError(SklearnMetric):
    """
    Compute absolute error regression loss

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 0.0, 2, 8])
        >>> y_true = torch.tensor([3, -0.5, 2, 7])
        >>> metric = MeanAbsoluteError()
        >>> metric(y_pred, y_true)
        tensor([0.5000])

    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'uniform_average',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’]
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_absolute_error',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean absolute error

        """
        return super().forward(y_true=y_true,
                               y_pred=y_pred,
                               sample_weight=sample_weight)


class MeanSquaredError(SklearnMetric):
    """
    Compute mean squared error loss

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 0.0, 2, 8])
        >>> y_true = torch.tensor([3, -0.5, 2, 7])
        >>> metric = MeanSquaredError()
        >>> metric(y_pred, y_true)
        tensor([0.3750])
        >>> metric = MeanSquaredError(squared=True)
        >>> metric(y_pred, y_true)
        tensor([0.6124])

    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'uniform_average',
            squared: bool = False,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’]
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            squared: if ``True`` returns the mse value else the rmse value
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_squared_error',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)
        self.squared = squared

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean squared error

        """
        mse = super().forward(y_true=y_true, y_pred=y_pred,
                              sample_weight=sample_weight)
        if self.squared:
            mse = np.sqrt(mse)
        return mse


class MeanSquaredLogError(SklearnMetric):
    """
    Calculates the mean squared log error

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 5, 4, 8])
        >>> y_true = torch.tensor([3, 5, 2.5, 7])
        >>> metric = MeanSquaredLogError()
        >>> metric(y_pred, y_true)
        tensor([0.0397])
    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'uniform_average',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’]
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_squared_log_error',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean squared log error

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)


class MedianAbsoluteError(SklearnMetric):
    """
    Calculates the median absolute error

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 0.0, 2, 8])
        >>> y_true = torch.tensor([3, -0.5, 2, 7])
        >>> metric = MedianAbsoluteError()
        >>> metric(y_pred, y_true)
        tensor([0.5000])
    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'uniform_average',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’]
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('median_absolute_error',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.

        Return:
            Median absolute error

        """
        return super().forward(y_true=y_true, y_pred=y_pred)


class R2Score(SklearnMetric):
    """
    Calculates the R^2 score also known as coefficient of determination

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2.5, 0.0, 2, 8])
        >>> y_true = torch.tensor([3, -0.5, 2, 7])
        >>> metric = R2Score()
        >>> metric(y_pred, y_true)
        tensor([0.9486])
    """

    def __init__(
            self,
            multioutput: Optional[Union[str, List[float]]] = 'uniform_average',
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            multioutput: either one of the strings [‘raw_values’, ‘uniform_average’, 'variance_weighted']
                or an array with shape (n_outputs,) that defines how multiple
                output values should be aggregated.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('r2_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         multioutput=multioutput)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            R^2 score

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)


class MeanPoissonDeviance(SklearnMetric):
    """
    Calculates the mean poisson deviance regression loss

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2, 0.5, 1, 4])
        >>> y_true = torch.tensor([0.5, 0.5, 2., 2.])
        >>> metric = MeanPoissonDeviance()
        >>> metric(y_pred, y_true)
        tensor([0.9034])
    """

    def __init__(
            self,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_poisson_deviance',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean possion deviance

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)


class MeanGammaDeviance(SklearnMetric):
    """
    Calculates the mean gamma deviance regression loss

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([0.5, 0.5, 2., 2.])
        >>> y_true = torch.tensor([2, 0.5, 1, 4])
        >>> metric = MeanGammaDeviance()
        >>> metric(y_pred, y_true)
        tensor([1.0569])
    """

    def __init__(
            self,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_gamma_deviance',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean gamma deviance

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)


class MeanTweedieDeviance(SklearnMetric):
    """
    Calculates the mean tweedie deviance regression loss

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code

    Example:

        >>> y_pred = torch.tensor([2, 0.5, 1, 4])
        >>> y_true = torch.tensor([0.5, 0.5, 2., 2.])
        >>> metric = MeanTweedieDeviance()
        >>> metric(y_pred, y_true)
        tensor([1.8125])
    """

    def __init__(
            self,
            power: float = 0,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            power: tweedie power parameter:

                * power < 0: Extreme stable distribution. Requires: y_pred > 0.
                * power = 0 : Normal distribution, output corresponds to mean_squared_error.
                    y_true and y_pred can be any real numbers.
                * power = 1 : Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
                * 1 < power < 2 : Compound Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
                * power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
                * power = 3 : Inverse Gaussian distribution. Requires: y_true > 0 and y_pred > 0.
                * otherwise : Positive stable distribution. Requires: y_true > 0 and y_pred > 0.

            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__('mean_tweedie_deviance',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         power=power)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            y_pred: Estimated target values
            y_true: Ground truth (correct) target values.
            sample_weight: Sample weights.

        Return:
            Mean tweedie deviance

        """
        return super().forward(y_true=y_true, y_pred=y_pred,
                               sample_weight=sample_weight)
