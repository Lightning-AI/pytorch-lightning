#!/usr/bin/env python
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

from __future__ import annotations

from collections import OrderedDict

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary as ModelSummaryCallback
from pytorch_lightning.utilities.model_summary import (
    _is_lazy_weight_tensor,
    get_human_readable_count,
    LayerSummary,
    ModelSummary,
)


def deepspeed_param_size(p: torch.nn.Parameter) -> int:
    assert hasattr(p, "ds_numel")
    return p.ds_numel


class DeepSpeedLayerSummary(LayerSummary):
    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters())

    @property
    def average_shard_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(p.partitioned_size() if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters())


class DeepSpeedSummary(ModelSummary):
    def summarize(self) -> dict[str, LayerSummary]:
        summary = OrderedDict((name, DeepSpeedLayerSummary(self._model)) for name, module in self.named_modules)
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
    def total_parameters(self) -> int:
        return sum(deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0
            for p in self._model.parameters()
            if p.requires_grad
        )

    @property
    def parameters_per_layer(self) -> list[int]:
        return [layer.average_shard_parameters for layer in self._layer_summary.values()]

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
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        return arrays


class DeepSpeedModelSummary(ModelSummaryCallback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._max_depth:
            return None
        from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy) and trainer.strategy.zero_stage_3:
            model_summary = DeepSpeedSummary(pl_module, max_depth=self._max_depth)
            summary_data = model_summary._get_summary_data()
            total_parameters = model_summary.total_parameters
            trainable_parameters = model_summary.trainable_parameters
            model_size = model_summary.model_size

            if trainer.is_global_zero:
                self.summarize(summary_data, total_parameters, trainable_parameters, model_size)
            return
        return super().on_fit_start(trainer, pl_module)
