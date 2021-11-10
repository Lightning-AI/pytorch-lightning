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
from typing import Any, Dict, Optional, Union

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import TPUPrecisionPlugin
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def setup(self, trainer: "pl.Trainer") -> None:
        """
        Raises:
            ValueError:
                If the precision or training type plugin are unsupported.
        """
        if not isinstance(self.precision_plugin, TPUPrecisionPlugin):
            # this configuration should have been avoided in the accelerator connector
            raise ValueError(
                f"The `TPUAccelerator` can only be used with a `TPUPrecisionPlugin`, found: {self.precision_plugin}."
            )
        if not isinstance(self.training_type_plugin, (SingleTPUPlugin, TPUSpawnPlugin)):
            raise ValueError(
                "The `TPUAccelerator` can only be used with a `SingleTPUPlugin` or `TPUSpawnPlugin,"
                f" found {self.training_type_plugin}."
            )
        return super().setup(trainer)

    def _move_optimizer_state(self, device: Optional[torch.device] = None) -> None:
        """Moves the state of the optimizers to the TPU if needed."""
        # TODO: `self.root_device` would raise error if called outside the spawn process
        # while training on 8 and more cores.
        for opt in self.optimizers:
            for p, v in opt.state.items():
                opt.state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, self.root_device)

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for the given TPU device.

        Args:
            device: TPU device for which to get stats

        Returns:
            A dictionary mapping the metrics (free memory and peak memory) to their values.
        """
        memory_info = xm.get_memory_info(device)
        free_memory = memory_info["kb_free"]
        peak_memory = memory_info["kb_total"] - free_memory
        device_stats = {
            "avg. free memory (MB)": free_memory,
            "avg. peak memory (MB)": peak_memory,
        }
        return device_stats

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 8
