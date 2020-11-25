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

import torch

from pytorch_lightning.metrics.utils import to_onehot, select_topk


def _check_shape_and_type_consistency(preds: torch.Tensor, target: torch.Tensor) -> Tuple[str, int]:
    """
    This checks that the shape and type of inputs are consistent with
    each other and fall into one of the allowed input types (see the
    documentation of docstring of _input_format_classification). It does
    not check for consistency of number of classes, other functions take
    care of that.

    It returns the name of the case in which the inputs fall, and the implied
    number of classes (from the C dim for multi-class data, or extra dim(s) for
    multi-label data).
    """

    preds_float = preds.is_floating_point()

    if preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "`preds` and `target` should have the same shape",
                f" got `preds shape = {preds.shape} and `target` shape = {target.shape}.",
            )
        if preds_float and target.max() > 1:
            raise ValueError(
                "if `preds` and `target` are of shape (N, ...) and `preds` are floats, `target` should be binary."
            )

        # Get the case
        if preds.ndim == 1 and preds_float:
            case = "binary"
        elif preds.ndim == 1 and not preds_float:
            case = "multi-class"
        elif preds.ndim > 1 and preds_float:
            case = "multi-label"
        else:
            case = "multi-dim multi-class"

        implied_classes = torch.prod(torch.Tensor(list(preds.shape[1:])))

    elif preds.ndim == target.ndim + 1:
        if not preds_float:
            raise ValueError("if `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if not preds.shape[:-1] == target.shape:
            if preds.shape[2:] != target.shape[1:]:
                raise ValueError(
                    "if `preds` have one dimension more than `target`, the shape of `preds` should be"
                    " either of shape (N, C, ...) or (N, ..., C), and of `target` of shape (N, ...)."
                )

        implied_classes = preds.shape[-1 if preds.shape[:-1] == target.shape else 1]

        if preds.ndim == 2:
            case = "multi-class"
        else:
            case = "multi-dim multi-class"
    else:
        raise ValueError(
            "`preds` and `target` should both have the (same) shape (N, ...), or `target` (N, ...)"
            " and `preds` (N, C, ...) or (N, ..., C)."
        )

    return case, implied_classes


def _check_num_classes_binary(num_classes: int, is_multiclass: bool):
    """
    This checks that the consistency of `num_classes` with the data
    and `is_multiclass` param for binary data.
    """

    if num_classes > 2:
        raise ValueError("Your data is binary, but `num_classes` is larger than 2.")
    elif num_classes == 2 and not is_multiclass:
        raise ValueError(
            "Your data is binary and `num_classes=2`, but `is_multiclass` is not True."
            " Set it to True if you want to transform binary data to multi-class format."
        )
    elif num_classes == 1 and is_multiclass:
        raise ValueError(
            "You have binary data and have set `is_multiclass=True`, but `num_classes` is 1."
            " Either leave `is_multiclass` unset or set it to 2 to transform binary data to multi-class format."
        )


def _check_num_classes_mc(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, is_multiclass: bool, implied_classes: int
):
    """
    This checks that the consistency of `num_classes` with the data
    and `is_multiclass` param for (multi-dimensional) multi-class data.
    """

    if num_classes == 1 and is_multiclass is not False:
        raise ValueError(
            "You have set `num_classes=1`, but predictions are integers."
            " If you want to convert (multi-dimensional) multi-class data with 2 classes"
            " to binary/multi-label, set `is_multiclass=False`."
        )
    elif num_classes > 1:
        if is_multiclass is False:
            if implied_classes != num_classes:
                raise ValueError(
                    "You have set `is_multiclass=False`, but the implied number of classes "
                    " (from shape of inputs) does not match `num_classes`. If you are trying to"
                    " transform multi-dim multi-class data with 2 classes to multi-label, `num_classes`"
                    " should be either None or the product of the size of extra dimensions (...)."
                    " See Input Types in Metrics documentation."
                )
        if num_classes <= target.max():
            raise ValueError("The highest label in `target` should be smaller than `num_classes`.")
        if num_classes <= preds.max():
            raise ValueError("The highest label in `preds` should be smaller than `num_classes`.")
        if preds.shape != target.shape and num_classes != implied_classes:
            raise ValueError("The size of C dimension of `preds` does not match `num_classes`.")


def _check_num_classes_ml(num_classes: int, is_multiclass: bool, implied_classes: int):
    """
    This checks that the consistency of `num_classes` with the data
    and `is_multiclass` param for multi-label data.
    """

    if is_multiclass and num_classes != 2:
        raise ValueError(
            "Your have set `is_multiclass=True`, but `num_classes` is not equal to 2."
            " If you are trying to transform multi-label data to 2 class multi-dimensional"
            " multi-class, you should set `num_classes` to either 2 or None."
        )
    if not is_multiclass and num_classes != implied_classes:
        raise ValueError("The implied number of classes (from shape of inputs) does not match num_classes.")


def _check_classification_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    num_classes: Optional[int] = None,
    is_multiclass: bool = False,
    top_k: int = 1,
) -> None:
    """Performs error checking on inputs for classification.

    This ensures that preds and target take one of the shape/type combinations that are
    specified in ``_input_format_classification`` docstring. It also checks the cases of
    over-rides with ``is_multiclass`` by checking (for multi-class and multi-dim multi-class
    cases) that there are only up to 2 distinct labels.

    In case where preds are floats (probabilities), it is checked whether they are in [0,1] interval.

    When ``num_classes`` is given, it is checked that it is consitent with input cases (binary,
    multi-label, ...), and that, if availible, the implied number of classes in the ``C``
    dimension is consistent with it (as well as that max label in target is smaller than it).

    When ``num_classes`` is not specified in these cases, consistency of the highest target
    value against ``C`` dimension is checked for (multi-dimensional) multi-class cases.

    If ``top_k`` is larger than one, then an error is raised if the inputs are not (multi-dim)
    multi-class with probability predictions.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (N).

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels, always integers
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
        num_classes: number of classes
        is_multiclass: if True, treat binary and multi-label inputs as multi-class or multi-dim
            multi-class with 2 classes, respectively. If False, treat multi-class and multi-dim
            multi-class inputs with 1 or 2 classes as binary and multi-label, respectively.
            Defaults to None, which treats inputs as they appear.
    """

    if target.is_floating_point():
        raise ValueError("`target` has to be an integer tensor.")
    elif target.min() < 0:
        raise ValueError("`target` has to be a non-negative tensor.")

    preds_float = preds.is_floating_point()
    if not preds_float and preds.min() < 0:
        raise ValueError("if `preds` are integers, they have to be non-negative.")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError("`preds` and `target` should have the same first dimension.")

    if preds_float:
        if preds.min() < 0 or preds.max() > 1:
            raise ValueError("`preds` should be probabilities, but values were detected outside of [0,1] range.")

    if threshold > 1 or threshold < 0:
        raise ValueError("`threshold` should be a probability in [0,1].")

    if is_multiclass is False and target.max() > 1:
        raise ValueError("If you set `is_multiclass=False`, then `target` should not exceed 1.")

    if is_multiclass is False and not preds_float and preds.max() > 1:
        raise ValueError("If you set `is_multiclass=False` and `preds` are integers, then `preds` should not exceed 1.")

    # Check that shape/types fall into one of the cases
    case, implied_classes = _check_shape_and_type_consistency(preds, target)

    if preds.shape != target.shape and is_multiclass is False and implied_classes != 2:
        raise ValueError(
            "You have set `is_multiclass=False`, but have more than 2 classes in your data,"
            " based on the C dimension of `preds`."
        )

    # Check that num_classes is consistent
    if not num_classes:
        if preds.shape != target.shape and target.max() >= implied_classes:
            raise ValueError("The highest label in `target` should be smaller than the size of C dimension.")
    else:
        if case == "binary":
            _check_num_classes_binary(num_classes, is_multiclass)
        elif "multi-class" in case:
            _check_num_classes_mc(preds, target, num_classes, is_multiclass, implied_classes)

        elif case == "multi-label":
            _check_num_classes_ml(num_classes, is_multiclass, implied_classes)

    # Check that if top_k > 1, we have (multi-class) multi-dim with probabilities
    if top_k > 1:
        if preds.shape == target.shape:
            raise ValueError(
                "You have set `top_k` above 1, but your data is not (multi-dimensional) multi-class"
                " with probability predictions."
            )


def _input_format_classification(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    top_k: int = 1,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Convert preds and target tensors into common format.

    Preds and targets are supposed to fall into one of these categories (and are
    validated to make sure this is the case):

    * Both preds and target are of shape ``(N,)``, and both are integers (multi-class)
    * Both preds and target are of shape ``(N,)``, and target is binary, while preds
        are a float (binary)
    * preds are of shape ``(N, C)`` and are floats, and target is of shape ``(N,)`` and
        is integer (multi-class)
    * preds and target are of shape ``(N, ...)``, target is binary and preds is a float
         (multi-label)
    * preds are of shape ``(N, ..., C)`` or ``(N, C, ...)`` and are floats, target is of
        shape ``(N, ...)`` and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
        multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``mode`` string, which describes which of the above cases the inputs belonged to - regardless
    of whether this was "overridden" by other settings (like ``is_multiclass``).

    In binary case, targets are normally returned as ``(N,1)`` tensor, while preds are transformed
    into a binary tensor (elements become 1 if the probability is greater than or equal to
    ``threshold`` or 0 otherwise). If ``is_multiclass=True``, then then both targets are preds
    become ``(N, 2)`` tensors by a one-hot transformation; with the thresholding being applied to
    preds first.

    In multi-class case, normally both preds and targets become ``(N, C)`` binary tensors; targets
    by a one-hot transformation and preds by selecting ``top_k`` largest entries (if their original
    shape was ``(N,C)``). However, if ``is_multiclass=False``, then targets and preds will be
    returned as ``(N,1)`` tensor.

    In multi-label case, normally targets and preds are returned as ``(N, C)`` binary tensors, with
    preds being binarized as in the binary case. Here the ``C`` dimension is obtained by flattening
    all dimensions after the first one. However if ``is_multiclass=True``, then both are returned as
    ``(N, 2, C)``, by an equivalent transformation as in the binary case.

    In multi-dimensional multi-class case, normally both target and preds are returned as
    ``(N, C, X)`` tensors, with ``X`` resulting from flattening of all dimensions except ``N`` and
    ``C``. The transformations performed here are equivalent to the multi-class case. However, if
    ``is_multiclass=False`` (and there are up to two classes), then the data is returned as
    ``(N, X)`` binary tensors (multi-label).

    Also, in multi-dimensional multi-class case, if the position of the ``C``
    dimension is ambiguous (e.g. if targets are a ``(7, 3)`` tensor, while predictions are a
    ``(7, 3, 3)`` tensor), it will be assumed that the ``C`` dimension is the second dimension.
    If this is not the case,  you should move it from the last to second place using
    ``torch.movedim(preds, -1, 1)``, or using ``preds.permute``, if you are using an older
    version of Pytorch.

    Note that where a one-hot transformation needs to be performed and the number of classes
    is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
    equal to ``num_classes``, if it is given, or the maximum label value in preds and
    target.

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels, always integers
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
        num_classes: number of classes
        top_k: number of highest probability entries for each sample to convert to 1s, relevant
            only for (multi-dimensional) multi-class cases.
        is_multiclass: if True, treat binary and multi-label inputs as multi-class or multi-dim
            multi-class with 2 classes, respectively. If False, treat multi-class and multi-dim
            multi-class inputs with 1 or 2 classes as binary and multi-label, respectively.
            Defaults to None, which treats inputs as they appear.

    Returns:
        preds: binary tensor of shape (N, C) or (N, C, X)
        target: binary tensor of shape (N, C) or (N, C, X)
    """
    preds, target = preds.clone().detach(), target.clone().detach()

    # Remove excess dimensions
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()

    _check_classification_inputs(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    preds_float = preds.is_floating_point()

    if preds.ndim == target.ndim == 1 and preds_float:
        mode = "binary"
        preds = (preds >= threshold).int()

        if is_multiclass:
            target = to_onehot(target, 2)
            preds = to_onehot(preds, 2)
        else:
            preds = preds.unsqueeze(-1)
            target = target.unsqueeze(-1)

    elif preds.ndim == target.ndim and preds_float:
        mode = "multi-label"
        preds = (preds >= threshold).int()

        if is_multiclass:
            preds = to_onehot(preds, 2).reshape(preds.shape[0], 2, -1)
            target = to_onehot(target, 2).reshape(target.shape[0], 2, -1)
        else:
            preds = preds.reshape(preds.shape[0], -1)
            target = target.reshape(target.shape[0], -1)

    elif preds.ndim == target.ndim + 1 == 2:
        mode = "multi-class"
        if not num_classes:
            num_classes = preds.shape[1]

        target = to_onehot(target, num_classes)
        preds = select_topk(preds, top_k)

        # If is_multiclass=False, force to binary
        if is_multiclass is False:
            target = target[:, [1]]
            preds = preds[:, [1]]

    elif preds.ndim == target.ndim == 1 and not preds_float:
        mode = "multi-class"

        if not num_classes:
            num_classes = max(preds.max(), target.max()) + 1

        # If is_multiclass=False, force to binary
        if is_multiclass is False:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)
        else:
            preds = to_onehot(preds, num_classes)
            target = to_onehot(target, num_classes)

    # Multi-dim multi-class (N, ...) with integers
    elif preds.shape == target.shape and not preds_float:
        mode = "multi-dim multi-class"

        if not num_classes:
            num_classes = max(preds.max(), target.max()) + 1

        # If is_multiclass=False, force to multi-label
        if is_multiclass is False:
            preds = preds.reshape(preds.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        else:
            target = to_onehot(target, num_classes)
            target = target.reshape(target.shape[0], target.shape[1], -1)
            preds = to_onehot(preds, num_classes)
            preds = preds.reshape(preds.shape[0], preds.shape[1], -1)

    # Multi-dim multi-class (N, C, ...) and (N, ..., C)
    else:
        mode = "multi-dim multi-class"
        if preds.shape[:-1] == target.shape:
            shape_permute = list(range(preds.ndim))
            shape_permute[1] = shape_permute[-1]
            shape_permute[2:] = range(1, len(shape_permute) - 1)

            preds = preds.permute(*shape_permute)

        num_classes = preds.shape[1]

        if is_multiclass is False:
            target = target.reshape(target.shape[0], -1)
            preds = select_topk(preds, 1)[:, 1, ...]
            preds = preds.reshape(preds.shape[0], -1)
        else:
            target = to_onehot(target, num_classes)
            target = target.reshape(target.shape[0], target.shape[1], -1)
            preds = select_topk(preds, top_k).reshape(preds.shape[0], preds.shape[1], -1)

    return preds, target, mode
