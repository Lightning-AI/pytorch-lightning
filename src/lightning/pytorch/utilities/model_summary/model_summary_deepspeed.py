# Copyright 2020 The PyTorch Lightning team and Microsoft Corporation. All rights reserved.
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
"""Utilities that can be used with Deepspeed."""

from collections import OrderedDict

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override

from lightning.pytorch.utilities.model_summary.model_summary import (
    NOT_APPLICABLE,
    LayerSummary,
    ModelSummary,
    _tensor_has_shape,
    get_human_readable_count,
)


def deepspeed_param_size(p: torch.nn.Parameter) -> int:
    assert hasattr(p, "ds_numel")
    return p.ds_numel


class DeepSpeedLayerSummary(LayerSummary):
    @property
    @override
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(deepspeed_param_size(p) if not _tensor_has_shape(p) else 0 for p in self._module.parameters())

    @property
    def average_shard_parameters(self) -> int:
        """Returns the number of parameters in this module."""

        def partitioned_size(p: Parameter) -> int:
            return p.partitioned_size() if RequirementCache("deepspeed<0.6.6") else p.partition_numel()

        return sum(partitioned_size(p) if not _tensor_has_shape(p) else 0 for p in self._module.parameters())


class DeepSpeedSummary(ModelSummary):
    @override
    def summarize(self) -> dict[str, DeepSpeedLayerSummary]:  # type: ignore[override]
        summary = OrderedDict((name, DeepSpeedLayerSummary(module)) for name, module in self.named_modules)
        if self._model.example_input_array is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()

        if self._max_depth >= 1:
            # remove summary entries with depth > max_depth
            for k in [k for k in summary if k.count(".") >= self._max_depth]:
                del summary[k]

        return summary

    @property
    @override
    def total_parameters(self) -> int:
        return sum(deepspeed_param_size(p) if not _tensor_has_shape(p) else 0 for p in self._model.parameters())

    @property
    @override
    def trainable_parameters(self) -> int:
        return sum(
            deepspeed_param_size(p) if not _tensor_has_shape(p) else 0
            for p in self._model.parameters()
            if p.requires_grad
        )

    @property
    def parameters_per_layer(self) -> list[int]:
        return [layer.average_shard_parameters for layer in self._layer_summary.values()]

    @override
    def _get_summary_data(self) -> list[tuple[str, list[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size

        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
            ("Params per Device", list(map(get_human_readable_count, self.parameters_per_layer))),
            ("Mode", ["train" if mode else "eval" for mode in self.training_modes]),
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)

        return arrays

    @override
    def _add_leftover_params_to_summary(self, arrays: list[tuple[str, list[str]]], total_leftover_params: int) -> None:
        """Add summary of params not associated with module or layer to model summary."""
        super()._add_leftover_params_to_summary(arrays, total_leftover_params)
        layer_summaries = dict(arrays)
        layer_summaries["Params per Device"].append(NOT_APPLICABLE)
