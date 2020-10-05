import math
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from collections.abc import Mapping, Sequence
from collections import namedtuple

import torch
from torch import nn
from pytorch_lightning.metrics.metric import Metric


class Accuracy(Metric):
    """
    Computes accuracy. Works with binary, multiclass, and multilabel data.

    preds and targets must be of shape (N, ...) and (N, ...) or (N, num_classes, ...) and (N, ...)

    If preds and targets are the same shape:
        If preds are integer values, we perform accuracy with those values
        If preds are floating point we threshold at `threshold`

    """
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)

        # change to dist_reduce_fx
        self.add_state("correct", torch.tensor(0), reduction=sum)
        self.add_state("total", torch.tensor(0), reduction=sum)

        self.threshold = threshold

    def _input_format(self, preds, target):
        if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
            raise ValueError(
                "preds and target must have same number of dimensions, or one additional dimension for preds"
            )

        if len(preds.shape) == len(target.shape) + 1:
            # multi class probabilites
            preds = torch.argmax(preds, dim=1)

        if len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
            # binary or multilabel probablities
            preds = (preds >= self.threshold).long()

        return preds, target

    def update(self, preds, target):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
