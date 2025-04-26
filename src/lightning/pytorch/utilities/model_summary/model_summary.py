# Copyright The Lightning AI team.
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
"""Utilities related to model weights summary."""

import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

import lightning.pytorch as pl
from lightning.fabric.utilities.distributed import _is_dtensor
from lightning.pytorch.utilities.model_helpers import _ModuleMode
from lightning.pytorch.utilities.rank_zero import WarningCache

log = logging.getLogger(__name__)
warning_cache = WarningCache()

PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
UNKNOWN_SIZE = "?"
LEFTOVER_PARAMS_NAME = "other params"
NOT_APPLICABLE = "n/a"


class LayerSummary:
    """Summary class for a single layer in a :class:`~lightning.pytorch.core.LightningModule`. It collects the
    following information:

    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Number of parameters

    The input and output shapes are only known after the example input array was
    passed through the model.

    Example::

        >>> model = torch.nn.Conv2d(3, 8, 3)
        >>> summary = LayerSummary(model)
        >>> summary.num_parameters
        224
        >>> summary.layer_type
        'Conv2d'
        >>> output = model(torch.rand(1, 3, 5, 5))
        >>> summary.in_size
        [1, 3, 5, 5]
        >>> summary.out_size
        [1, 8, 3, 3]

    Args:
        module: A module to summarize

    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self._in_size: Optional[Union[str, list]] = None
        self._out_size: Optional[Union[str, list]] = None

    def __del__(self) -> None:
        self.detach_hook()

    def _register_hook(self) -> Optional[RemovableHandle]:
        """Registers a hook on the module that computes the input- and output size(s) on the first forward pass. If the
        hook is called, it will remove itself from the from the module, meaning that recursive models will only record
        their input- and output shapes once. Registering hooks on :class:`~torch.jit.ScriptModule` is not supported.

        Return:
            A handle for the installed hook, or ``None`` if registering the hook is not possible.

        """

        def hook(_: nn.Module, inp: Any, out: Any) -> None:
            if len(inp) == 1:
                inp = inp[0]

            self._in_size = parse_batch_shape(inp)
            self._out_size = parse_batch_shape(out)
            assert self._hook_handle is not None
            self._hook_handle.remove()

        def hook_with_kwargs(_: nn.Module, args: Any, kwargs: Any, out: Any) -> None:
            # We can't write them in the same function, since the forward hook
            # uses positional arguments.

            inp = (*args, *kwargs.values()) if kwargs is not None else args
            hook(_, inp, out)

        handle = None
        if not isinstance(self._module, torch.jit.ScriptModule):
            handle = self._module.register_forward_hook(hook_with_kwargs, with_kwargs=True)

        return handle

    def detach_hook(self) -> None:
        """Removes the forward hook if it was not already removed in the forward pass.

        Will be called after the summary is created.

        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @property
    def in_size(self) -> Union[str, list]:
        return self._in_size or UNKNOWN_SIZE

    @property
    def out_size(self) -> Union[str, list]:
        return self._out_size or UNKNOWN_SIZE

    @property
    def layer_type(self) -> str:
        """Returns the class name of the module."""
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(p.numel() if not _tensor_has_shape(p) else 0 for p in self._module.parameters())

    @property
    def training(self) -> bool:
        """Returns whether the module is in training mode."""
        return self._module.training


class ModelSummary:
    """Generates a summary of all layers in a :class:`~lightning.pytorch.core.LightningModule`.

    Args:
        model: The model to summarize (also referred to as the root module).

        max_depth: Maximum depth of modules to show. Use -1 to show all modules or 0 to show no
            summary. Defaults to 1.

    The string representation of this summary prints a table with columns containing
    the name, type and number of parameters for each layer.

    The root module may also have an attribute ``example_input_array`` as shown in the example below.
    If present, the root module will be called with it as input to determine the
    intermediate input- and output shapes of all layers. Supported are tensors and
    nested lists and tuples of tensors. All other types of inputs will be skipped and show as `?`
    in the summary table. The summary will also display `?` for layers not used in the forward pass.
    If there are parameters not associated with any layers or modules, the count of those parameters
    will be displayed in the table under `other params`. The summary will display `n/a` for module type,
    in size, and out size.

    Example::

        >>> import lightning.pytorch as pl
        >>> class LitModel(pl.LightningModule):
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512))
        ...         self.example_input_array = torch.zeros(10, 256)  # optional
        ...
        ...     def forward(self, x):
        ...         return self.net(x)
        ...
        >>> model = LitModel()
        >>> ModelSummary(model, max_depth=1)  # doctest: +NORMALIZE_WHITESPACE
          | Name | Type       | Params | Mode  | In sizes  | Out sizes
        --------------------------------------------------------------------
        0 | net  | Sequential | 132 K  | train | [10, 256] | [10, 512]
        --------------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
        3         Modules in train mode
        0         Modules in eval mode
        >>> ModelSummary(model, max_depth=-1)  # doctest: +NORMALIZE_WHITESPACE
          | Name  | Type        | Params | Mode  | In sizes  | Out sizes
        ----------------------------------------------------------------------
        0 | net   | Sequential  | 132 K  | train | [10, 256] | [10, 512]
        1 | net.0 | Linear      | 131 K  | train | [10, 256] | [10, 512]
        2 | net.1 | BatchNorm1d | 1.0 K  | train | [10, 512] | [10, 512]
        ----------------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
        3         Modules in train mode
        0         Modules in eval mode

    """

    def __init__(self, model: "pl.LightningModule", max_depth: int = 1) -> None:
        self._model = model

        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")

        self._max_depth = max_depth
        self._layer_summary = self.summarize()
        # 1 byte -> 8 bits
        # TODO: how do we compute precision_megabytes in case of mixed precision?
        precision_to_bits = {"64": 64, "32": 32, "16": 16, "bf16": 16}
        precision = precision_to_bits.get(self._model.trainer.precision, 32) if self._model._trainer else 32
        self._precision_megabytes = (precision / 8.0) * 1e-6

    @property
    def named_modules(self) -> list[tuple[str, nn.Module]]:
        mods: list[tuple[str, nn.Module]]
        if self._max_depth == 0:
            mods = []
        elif self._max_depth == 1:
            # the children are the top-level modules
            mods = list(self._model.named_children())
        else:
            mods = self._model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        return mods

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> list[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> list:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> list:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> list[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def training_modes(self) -> list[bool]:
        return [layer.training for layer in self._layer_summary.values()]

    @property
    def total_training_modes(self) -> dict[str, int]:
        modes = [layer.training for layer in self._model.modules()]
        modes = modes[1:]  # exclude the root module
        return {"train": modes.count(True), "eval": modes.count(False)}

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() if not _tensor_has_shape(p) else 0 for p in self._model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() if not _tensor_has_shape(p) else 0 for p in self._model.parameters() if p.requires_grad)

    @property
    def total_layer_params(self) -> int:
        return sum(self.param_nums)

    @property
    def model_size(self) -> float:
        return self.total_parameters * self._precision_megabytes

    def summarize(self) -> dict[str, LayerSummary]:
        summary = OrderedDict((name, LayerSummary(module)) for name, module in self.named_modules)
        if self._model.example_input_array is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()

        if self._max_depth >= 1:
            # remove summary entries with depth > max_depth
            for k in [k for k in summary if k.count(".") >= self._max_depth]:
                del summary[k]

        return summary

    def _forward_example_input(self) -> None:
        """Run the example input through each layer to get input- and output sizes."""
        model = self._model
        # the summary is supported without a trainer instance so we need to use the underscore property
        trainer = self._model._trainer

        input_ = model.example_input_array
        input_ = model._on_before_batch_transfer(input_)
        input_ = model._apply_batch_transfer_handler(input_)

        mode = _ModuleMode()
        mode.capture(model)
        model.eval()

        forward_context = contextlib.nullcontext() if trainer is None else trainer.precision_plugin.forward_context()
        with torch.no_grad(), forward_context:
            # let the model hooks collect the input- and output shapes
            if isinstance(input_, (list, tuple)):
                model(*input_)
            elif isinstance(input_, dict):
                model(**input_)
            else:
                model(input_)
        mode.restore(model)

    def _get_summary_data(self) -> list[tuple[str, list[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size

        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
            ("Mode", ["train" if mode else "eval" for mode in self.training_modes]),
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)

        return arrays

    def _add_leftover_params_to_summary(self, arrays: list[tuple[str, list[str]]], total_leftover_params: int) -> None:
        """Add summary of params not associated with module or layer to model summary."""
        layer_summaries = dict(arrays)
        layer_summaries[" "].append(" ")
        layer_summaries["Name"].append(LEFTOVER_PARAMS_NAME)
        layer_summaries["Type"].append(NOT_APPLICABLE)
        layer_summaries["Params"].append(get_human_readable_count(total_leftover_params))
        layer_summaries["Mode"].append(NOT_APPLICABLE)
        if "In sizes" in layer_summaries:
            layer_summaries["In sizes"].append(NOT_APPLICABLE)
        if "Out sizes" in layer_summaries:
            layer_summaries["Out sizes"].append(NOT_APPLICABLE)

    def __str__(self) -> str:
        arrays = self._get_summary_data()

        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters
        model_size = self.model_size
        total_training_modes = self.total_training_modes

        return _format_summary_table(total_parameters, trainable_parameters, model_size, total_training_modes, *arrays)

    def __repr__(self) -> str:
        return str(self)


def parse_batch_shape(batch: Any) -> Union[str, list]:
    if hasattr(batch, "shape"):
        return list(batch.shape)

    if isinstance(batch, (list, tuple)):
        return [parse_batch_shape(el) for el in batch]

    return UNKNOWN_SIZE


def _format_summary_table(
    total_parameters: int,
    trainable_parameters: int,
    model_size: float,
    total_training_modes: dict[str, int],
    *cols: tuple[str, list[str]],
) -> str:
    """Takes in a number of arrays, each specifying a column in the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted."""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width

    summary += "\n" + s.format(get_human_readable_count(trainable_parameters), 10)
    summary += "Trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters - trainable_parameters), 10)
    summary += "Non-trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters), 10)
    summary += "Total params"
    summary += "\n" + s.format(get_formatted_model_size(model_size), 10)
    summary += "Total estimated model params size (MB)"
    summary += "\n" + s.format(total_training_modes["train"], 10)
    summary += "Modules in train mode"
    summary += "\n" + s.format(total_training_modes["eval"], 10)
    summary += "Modules in eval mode"

    return summary


def get_formatted_model_size(total_model_size: float) -> str:
    return f"{total_model_size:,.3f}"


def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions, billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(math.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def _tensor_has_shape(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    # DTensor is a subtype of `UninitializedParameter`, but the shape is known
    if isinstance(p, UninitializedParameter) and not _is_dtensor(p):
        warning_cache.warn(
            "The total number of parameters detected may be inaccurate because the model contains"
            " an instance of `UninitializedParameter`. To get an accurate number, set `self.example_input_array`"
            " in your LightningModule."
        )
        return True
    return False


def summarize(lightning_module: "pl.LightningModule", max_depth: int = 1) -> ModelSummary:
    """Summarize the LightningModule specified by `lightning_module`.

    Args:
        lightning_module: `LightningModule` to summarize.

        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off. Default: 1.

    Return:
        The model summary object

    """
    return ModelSummary(lightning_module, max_depth=max_depth)
