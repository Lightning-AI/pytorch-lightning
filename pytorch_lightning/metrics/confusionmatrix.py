import numpy as np
from pytorch_lightning.metrics.metric import NumpyMetric


class ConfusionMatrix(NumpyMetric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, ignore_label):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def __call__(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (numpy.ndarray): Can be an N x K array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-array of integer values between 0 and K-1.
        - target (numpy.ndarray): Can be an N x K array of
        ground-truth classes for N examples and K classes, or an N-array
        of integer values between 0 and K-1.

        """
        _, predicted = predicted.max(1)

        predicted = predicted.view(-1)
        target = target.view(-1)

        ind = ~np.isin(target, self.ignore_label)
        predicted, target = predicted[ind], target[ind]

        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"

        if np.ndim(predicted) != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (
                predicted.min() >= 0
            ), "predicted values are not between 0 and k-1"

        if np.ndim(target) != 1:
            assert (
                target.shape[1] == self.num_classes
            ), "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (
                target <= 1
            ).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (target.sum(1) == 1).all(), "multi-label setting is not supported"
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (
                target.min() >= 0
            ), "target values are not between 0 and k-1"

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def get_confusion_matrix(self, normalized=False):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        return self.conf

    def get_iou(self):
        """Computes the intersection over union (IoU) per class and corresponding
        mean (mIoU).

        Intersection over union (IoU) is a common evaluation metric for semantic
        segmentation. The predictions are first accumulated in a confusion matrix
        and the IoU is computed from it as follows:

            IoU = true_positive / (true_positive + false_positive + false_negative).

        Keyword arguments:
        - num_classes (int): number of classes in the classification problem
        - normalized (boolean, optional): Determines whether or not the confusion
        matrix is normalized or not. Default: False.
        - ignore_index (int or iterable, optional): Index of the classes to ignore
        when computing the IoU. Can be an int, or any iterable of ints.

        Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter

        Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.

        """
        conf_matrix = self.get_confusion_matrix()
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)
            precision = true_positive / (true_positive + false_negative)
            overall_precision = np.sum(true_positive) / np.sum(conf_matrix)

        return {
            "iou": iou,
            "mean_iou": np.nanmean(iou),
            "precision_per_class": precision,
            "mean_precision": np.nanmean(precision),
            "overall_precision": overall_precision,
        }
