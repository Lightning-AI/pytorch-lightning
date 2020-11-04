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
from typing import Tuple, Optional

import numpy as np
import torch

from pytorch_lightning.metrics.utils import to_onehot


def _mask_zeros(tensor: torch.Tensor):
    """ Mask zeros in a tensor with 1.0. """

    ones = torch.ones(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    return torch.where(tensor == 0, ones, tensor)


def _check_classification_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    num_classes: Optional[int] = None,
    logits: bool = False,
) -> None:
    """Performs error checking on inputs for classification.

    This ensures that preds and target take one of the following shape/type combinations:

    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``. and target is binary, while
        preds are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and is
        integer (multi-class)
    * preds and target are of shape ``(N, C)``, target is binary and preds is a float
        (multi-label)
    * preds are of shape ``(N, ..., C)`` or ``(N, C, ...)`` and are floats, target is of shape
        ``(N, ...)`` and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
        multi-class)

    Additionally, in case where preds are floats and ``logits=False`` (so preds are
    probabilities), it is checked whether they are in [0,1] interval.

    If ``num_classes`` is specified, it is checked if the implied number of classes (by the
    ``C`` dimension) is equal to it. Similarly, if preds or target are integers, it is
    checked whether their max value is lower than ``C`` or ``num_classes``, if either
    exists/is specified.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (N).

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels, always integers
        threshold: probability (float) used for thresholding binary and multilabel input
        num_classes: number of classes
        logits: whether predictions are logits (before sigmoid) or probabilities (after sigmoid).
            Relevant when preds are floats
    """

    if target.is_floating_point():
        raise ValueError("target has to be an integer tensor")
    elif target.min() < 0:
        raise ValueError("target has to be a non-negative tensor")

    if not preds.is_floating_point() and preds.min() < 0:
        raise ValueError("if preds are integers, they have to be non-negative")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError("preds and target should have the same first dimension.")

    # If preds are probabilities (logits=False), check that they fall in [0,1]
    if preds.is_floating_point() and not logits:
        if preds.min() < 0 or preds.max() > 1:
            raise ValueError(
                "preds should be probabilities (logits=False), but values were detected outside of [0,1] range"
            )

    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold should be a probability in [0,1]")

    # Check that shape/types fall into one of the cases
    if len(preds.shape) == len(target.shape) == 1:
        if preds.is_floating_point() and target.max() > 1:
            raise ValueError("if preds are of shape (N,) and floats, target tensor should be binary")

    elif len(preds.shape) == len(target.shape) + 1:
        if not preds.is_floating_point():
            raise ValueError("if preds have one dimension more than target, preds should be a float tensor")
        if not preds.shape[:-1] == target.shape:
            if preds.shape[2:] != target.shape[1:]:
                raise ValueError(
                    "if preds if preds have one dimension more than target, the shape of preds should be"
                    "either (N, C, ...) or (N, ..., C), and of targets (N, ...)"
                )

    elif len(preds.shape) == len(target.shape):
        if preds.shape != target.shape:
            raise ValueError("if preds and target have the same number of dimensions, they should have the same shape")
        if len(preds.shape) == 2 and preds.is_floating_point() and target.max() > 1:
            raise ValueError(
                "if preds and target have 2 dimensions and preds are a float tensor, target should be binary"
            )
        if preds.is_floating_point() and len(preds.shape) > 2:
            raise ValueError(
                "When preds and target have the same shape and more than 2 dimensions, preds have to be integers"
            )

    else:
        raise ValueError(
            "The shapes of preds and target are not allowed, consult the metric documentation for allowed shapes."
        )

    # Check that number of classes is consistent
    if num_classes:
        if num_classes > 1:
            if target.max() >= num_classes:
                raise ValueError("Maximum label in target must be smaller than num_classes")
            if not preds.is_floating_point() and preds.max() >= num_classes:
                raise ValueError("Maximum label in preds must be smaller than num_classes")
            # multi-label or multi-class
            if preds.is_floating_point() and (preds.shape == target.shape or len(preds.shape) == 2):
                if preds.shape[1] != num_classes:
                    raise ValueError("Second dimension of preds does not match num_classes")
            # multi-dim multi-class
            elif preds.is_floating_point():
                extra_dim = -1 if preds.shape[:-1] == target.shape else 1
                if preds.shape[extra_dim] != num_classes:
                    raise ValueError(f"Extra ({extra_dim}) dimension of preds does not match num_classes")
        # Binary
        else:
            if target.max() > 1:
                raise ValueError("Maximum label in target is larger than 1")
            # As inputs are supposed to be squeezed, this is only the case if preds.shape[1] > 1
            if len(preds.shape) > 1:
                raise ValueError("If num_classes=1, preds must be a 1d tensor")
            if len(target.shape) > 1:
                raise ValueError("If num_classes=1, target must be a 1d tensor")
    else:
        # multi label and (multi-dim) multi class
        if preds.is_floating_point():
            extra_dim = -1 if preds.shape[:-1] == target.shape else 1
            implied_num_classes = preds.shape[extra_dim]
            if target.max() > implied_num_classes:
                raise ValueError(
                    "Maximum label in target is larger than the number of classes implied by the preds vector shape"
                )


def _input_format_classification(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    num_classes: Optional[int] = None,
    logits: bool = False,
    top_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert preds and target tensors into common format.

    Preds and targets are supposed to fall into one of these categories (inputs are
    validated to make sure this is the case):

    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``. and target is binary, while
        preds are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and is
        integer (multi-class)
    * preds and target are of shape ``(N, C)``, target is binary and preds is a float
        (multi-label)
    * preds are of shape ``(N, ..., C)`` or ``(N, C, ...)`` and are floats, target is of shape
        ``(N, ...)`` and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
        multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    In binary, multi-class and multi-label cases preds and target are transformed into
    ``(N, C)`` binary tensors. If labels need to be transformed to this shape a simple 
    one-hot transformation is applied, while for the preds vector this depends on the case:

    * In binary case the
    * In multi-class case where preds are ``(N,)`` tensor, a one-hot transformation 
        is applied
    * In multi-class case where preds are ``(N,C)`` tensor, ``top_k`` entries with the
        highest probability are labeled as 1, others as 0.
    * In multi-label  
    
    In binary case ``C=1`` by default, unless ``num_classes=2``
    is specified. In case when if preds are floats (logits/probabilities), the ``top_k``
    largest probabilities for each sample will become 1 in the final output.

    For multi-dimensional multi-class case where preds are probabilities/logits, the preds
    vector is transformed into a ``(N, ...)`` tensor, where entries are labels with the
    highest probability. Note that ``top_k`` argument is ignored in this case 

    Additionally, in case where preds are floats and ``logits=False`` (so preds are
    probabilities), it is checked whether they are in [0,1] interval.

    If ``num_classes`` is specified, it is checked if the implied number of classes (by the
    ``C`` dimension) is equal to it. Similarly, if preds or target are integers, it is
    checked whether their max value is lower than ``C`` or ``num_classes``, if either
    exists/is specified.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (N).

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels, always integers
        threshold: probability (float) used for thresholding binary and multilabel input
        num_classes: number of classes
        logits: whether predictions are logits (before sigmoid) or probabilities (after sigmoid).
            Relevant when preds are floats
    * binary classification: both preds and targets are tensors of shape (N,)
    * multi-class classification: targets are a tensor of shape (N,), preds are a
        tensor (of logits or probabilities) of shape (N, C)
    * multi-label classification: both targets and preds are of shape (N, ...).
        All dimensions after the first will be flattened down.

    All excess dimensions (of size 1, except for first) are squeezed at the beginning.

    Args:
        preds: either tensor with labels, tensor with probabilities/logits or
            multilabel tensor
        target: tensor with ground true labels
        threshold: probability (float) used for thresholding binary and multilabel input
        num_classes: number of classes in case of multi-class classification
        logits: whether predictions are logits (before sigmoid) or probabilities (after sigmoid).
            Relevant only for binary and multi-label classification

    Returns:
        preds: tensor (N, C) with labels
        target: tensor (N, C) with labels
    """
    # Remove excess dimensions
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()

    # TEMPORARY - "multidim" multiclass
    if len(preds.shape) > 2 and len(preds.shape) > len(target.shape):
        preds = torch.transpose(preds, 0, 1)
        preds = preds.reshape(preds.shape[0], -1)
        preds = torch.transpose(preds, 0, 1)
        target = target.reshape(-1)

    # Check that inputs are valid
    _check_classification_inputs(preds, target, threshold, num_classes, logits)

    is_multiclass = len(preds.shape) > len(target.shape) or (len(preds.shape) == 1 and not preds.is_floating_point())

    if logits:
        threshold = np.log(threshold / (1 - threshold))

    # binary or multilabel classification
    if not is_multiclass:
        preds, target = preds.view(preds.shape[0], -1), target.view(target.shape[0], -1)
        if preds.is_floating_point():
            preds = (preds >= threshold).to(torch.int)

    # multi-class classification, preds are probabilities
    elif len(preds.shape) == len(target.shape) + 1:
        num_classes = preds.shape[1]
        preds = torch.argmax(preds, dim=1)
        preds, target = to_onehot(preds, num_classes), to_onehot(target, num_classes)

    # multi-class classification, preds are labels
    else:
        if not num_classes:
            num_classes = max(preds.max(), target.max()) + 1
            num_classes = 1 if num_classes == 2 else num_classes

        # If labels/target are not binary (or are binary and user specifically wants 2 classes)
        if num_classes > 1:
            preds, target = to_onehot(preds, num_classes), to_onehot(target, num_classes)
        else:
            preds, target = preds.view(preds.shape[0], -1), target.view(target.shape[0], -1)

    return preds, target


def _stat_scores(
    preds: torch.Tensor, target: torch.Tensor, average: str = "micro"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the number of tp,fp,tn,fn.

    Parameters
    ----------
    labels
        A `[n_samples, n_labels]` tensor of true labels (0 or 1)
    preds
        A `[n_samples, n_labels]` tensor of predictions (0 or 1)

    Returns
    -------
    tp, fp, tn, fn
    """
    if average in ["binary", "micro"]:
        dim = list(range(len(preds.shape)))
    elif average in ["macro", "weighted", "none", "samples"]:
        dim = 1 if average == "samples" else 0

    true_pred, false_pred = target == preds, target != preds

    tp = (true_pred * (preds == 1)).sum(dim=dim)
    fp = (false_pred * (preds == 1)).sum(dim=dim)

    tn = (true_pred * (preds == 0)).sum(dim=dim)
    fn = (false_pred * (preds == 0)).sum(dim=dim)

    return tp, fp, tn, fn
