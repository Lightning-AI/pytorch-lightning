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


def _basic_input_validation(preds: torch.Tensor, target: torch.Tensor, threshold: float, is_multiclass: bool):
    """
    Perform basic validation of inputs that does not require deducing any information
    of the type of inputs.
    """

    if target.is_floating_point():
        raise ValueError("The `target` has to be an integer tensor.")
    if target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")

    preds_float = preds.is_floating_point()
    if not preds_float and preds.min() < 0:
        raise ValueError("If `preds` are integers, they have to be non-negative.")

    if not preds.shape[0] == target.shape[0]:
        raise ValueError("The `preds` and `target` should have the same first dimension.")

    if preds_float and (preds.min() < 0 or preds.max() > 1):
        raise ValueError("The `preds` should be probabilities, but values were detected outside of [0,1] range.")

    if threshold > 1 or threshold < 0:
        raise ValueError("The `threshold` should be a probability in [0,1].")

    if is_multiclass is False and target.max() > 1:
        raise ValueError("If you set `is_multiclass=False`, then `target` should not exceed 1.")

    if is_multiclass is False and not preds_float and preds.max() > 1:
        raise ValueError("If you set `is_multiclass=False` and `preds` are integers, then `preds` should not exceed 1.")


def _check_shape_and_type_consistency(preds: torch.Tensor, target: torch.Tensor) -> Tuple[str, int]:
    """
    This checks that the shape and type of inputs are consistent with
    each other and fall into one of the allowed input types (see the
    documentation of docstring of ``_input_format_classification``). It does
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
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        if preds_float and target.max() > 1:
            raise ValueError(
                "If `preds` and `target` are of shape (N, ...) and `preds` are floats, `target` should be binary."
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

        implied_classes = preds[0].numel()

    elif preds.ndim == target.ndim + 1:
        if not preds_float:
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )

        implied_classes = preds.shape[1]

        if preds.ndim == 2:
            case = "multi-class"
        else:
            case = "multi-dim multi-class"
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    return case, implied_classes


def _check_num_classes_binary(num_classes: int, is_multiclass: bool):
    """
    This checks that the consistency of `num_classes` with the data
    and `is_multiclass` param for binary data.
    """

    if num_classes > 2:
        raise ValueError("Your data is binary, but `num_classes` is larger than 2.")
    if num_classes == 2 and not is_multiclass:
        raise ValueError(
            "Your data is binary and `num_classes=2`, but `is_multiclass` is not True."
            " Set it to True if you want to transform binary data to multi-class format."
        )
    if num_classes == 1 and is_multiclass:
        raise ValueError(
            "You have binary data and have set `is_multiclass=True`, but `num_classes` is 1."
            " Either set `is_multiclass=None`(default) or set `num_classes=2` to transform binary data to multi-class format."
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
    if num_classes > 1:
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


def _check_top_k(top_k: int, case: str, implied_classes: int, is_multiclass: Optional[bool], preds_float: bool):
    if "multi-class" not in case or not preds_float:
        raise ValueError(
            "You have set `top_k` above 1, but your data is not (multi-dimensional) multi-class"
            " with probability predictions."
        )
    if is_multiclass is False:
        raise ValueError("If you set `is_multiclass=False`, you can not set `top_k`.")
    if top_k >= implied_classes:
        raise ValueError("The `top_k` has to be strictly smaller than the `C` dimension of `preds`.")


def _check_classification_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    num_classes: Optional[int],
    is_multiclass: bool,
    top_k: Optional[int],
) -> str:
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

    If ``top_k`` is set (not None) for inputs which are not (multi-dimensional) multi class
    with probabilities, then an error is raised. Similarly if ``top_k`` is set to a number
    that is higher than or equal to the ``C`` dimension of ``preds``.

    Preds and target tensors are expected to be squeezed already - all dimensions should be
    greater than 1, except perhaps the first one (N).

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be infered
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interepreted as 1 for these inputs.

            Should be left unset (``None``) for all other types of inputs.
        is_multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be (see :ref:`metrics: Input types` documentation section for
            input classification and examples of the use of this parameter). Should be left at default
            value (``None``) in most cases.

            The special cases where this parameter should be set are:

            - When you want to treat binary or multi-label inputs as multi-class or multi-dimensional
              multi-class with 2 classes, respectively. The probabilities are interpreted as the
              probability of the "1" class, and thresholding still applies as usual. In this case
              the parameter should be set to ``True``.
            - When you want to treat multi-class or multi-dimensional mulit-class inputs with 2 classes
              as binary or multi-label inputs, respectively. This is mainly meant for the case when
              inputs are labels, but will work if they are probabilities as well. For this case the
              parameter should be set to ``False``.

    Return:
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'
    """

    # Baisc validation (that does not need case/type information)
    _basic_input_validation(preds, target, threshold, is_multiclass)

    # Check that shape/types fall into one of the cases
    case, implied_classes = _check_shape_and_type_consistency(preds, target)

    # For (multi-dim) multi-class case with prob preds, check that preds sum up to 1
    if "multi-class" in case and preds.is_floating_point():
        if not torch.isclose(preds.sum(dim=1), torch.ones_like(preds.sum(dim=1))).all():
            raise ValueError("Probabilities in `preds` must sum up to 1 accross the `C` dimension.")

    # Check consistency with the `C` dimension in case of multi-class data
    if preds.shape != target.shape:
        if is_multiclass is False and implied_classes != 2:
            raise ValueError(
                "You have set `is_multiclass=False`, but have more than 2 classes in your data,"
                " based on the C dimension of `preds`."
            )
        if target.max() >= implied_classes:
            raise ValueError(
                "The highest label in `target` should be smaller than the size of the `C` dimension of `preds`."
            )

    # Check that num_classes is consistent
    if num_classes:
        if case == "binary":
            _check_num_classes_binary(num_classes, is_multiclass)
        elif "multi-class" in case:
            _check_num_classes_mc(preds, target, num_classes, is_multiclass, implied_classes)
        elif case == "multi-label":
            _check_num_classes_ml(num_classes, is_multiclass, implied_classes)

    # Check that top_k is consistent
    if top_k:
        _check_top_k(top_k, case, implied_classes, is_multiclass, preds.is_floating_point())

    return case


def _input_format_classification(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
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
    * preds are of shape ``(N, C, ...)`` and are floats, target is of shape ``(N, ...)``
      and is integer (multi-dimensional multi-class)
    * preds and target are of shape ``(N, ...)`` both are integers (multi-dimensional
      multi-class)

    To avoid ambiguities, all dimensions of size 1, except the first one, are squeezed out.

    The returned output tensors will be binary tensors of the same shape, either ``(N, C)``
    of ``(N, C, X)``, the details for each case are described below. The function also returns
    a ``case`` string, which describes which of the above cases the inputs belonged to - regardless
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

    Note that where a one-hot transformation needs to be performed and the number of classes
    is not implicitly given by a ``C`` dimension, the new ``C`` dimension will either be
    equal to ``num_classes``, if it is given, or the maximum label value in preds and
    target.

    Args:
        preds: Tensor with predictions (labels or probabilities)
        target: Tensor with ground truth labels, always integers (labels)
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs. Default: 0.5
        num_classes:
            Number of classes. If not explicitly set, the number of classes will be infered
            either from the shape of inputs, or the maximum label in the ``target`` and ``preds``
            tensor, where applicable.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interepreted as 1 for these inputs.

            Should be left unset (``None``) for all other types of inputs.
        is_multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be (see :ref:`metrics: Input types` documentation section for
            input classification and examples of the use of this parameter). Should be left at default
            value (``None``) in most cases.

            The special cases where this parameter should be set are:

            - When you want to treat binary or multi-label inputs as multi-class or multi-dimensional
              multi-class with 2 classes, respectively. The probabilities are interpreted as the
              probability of the "1" class, and thresholding still applies as usual. In this case
              the parameter should be set to ``True``.
            - When you want to treat multi-class or multi-dimensional mulit-class inputs with 2 classes
              as binary or multi-label inputs, respectively. This is mainly meant for the case when
              inputs are labels, but will work if they are probabilities as well. For this case the
              parameter should be set to ``False``.


    Returns:
        preds: binary tensor of shape (N, C) or (N, C, X)
        target: binary tensor of shape (N, C) or (N, C, X)
        case: The case the inputs fall in, one of 'binary', 'multi-class', 'multi-label' or
            'multi-dim multi-class'
    """
    # Remove excess dimensions
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()

    case = _check_classification_inputs(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    top_k = top_k if top_k else 1

    if case in ["binary", "multi-label"]:
        preds = (preds >= threshold).int()
        num_classes = num_classes if not is_multiclass else 2

    if "multi-class" in case or is_multiclass:
        if preds.is_floating_point():
            num_classes = preds.shape[1]
            preds = select_topk(preds, top_k)
        else:
            num_classes = num_classes if num_classes else max(preds.max(), target.max()) + 1
            preds = to_onehot(preds, max(2,num_classes))

        target = to_onehot(target, max(2,num_classes))

        if is_multiclass is False:
            preds, target = preds[:, 1, ...], target[:, 1, ...]

    if ("multi-class" in case and is_multiclass is not False) or is_multiclass:
        target = target.reshape(target.shape[0], target.shape[1], -1)
        preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
    else:
        target = target.reshape(target.shape[0], -1)
        preds = preds.reshape(preds.shape[0], -1)

    # Some operatins above create an extra dimension for MC/binary case - this removes it
    if preds.ndim > 2:
        preds, target = preds.squeeze(-1), target.squeeze(-1)

    return preds.int(), target.int(), case
