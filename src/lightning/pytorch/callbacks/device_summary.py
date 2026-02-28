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
"""
Device Summary
==============

Logs information about available and used devices (GPU, TPU, etc.) at the start of training.

"""

from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


class DeviceSummary(Callback):
    r"""Logs information about available and used devices at the start of training.

    This callback prints the availability and usage status of GPUs (CUDA/MPS) and TPUs.
    It also warns if a device is available but not being used.

    Args:
        show_warnings: Whether to show warnings when available devices are not used.
            Defaults to ``True``.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import DeviceSummary
        >>> # Default behavior - shows device info and warnings
        >>> trainer = Trainer(callbacks=[DeviceSummary()])
        >>> # Suppress device availability warnings
        >>> trainer = Trainer(callbacks=[DeviceSummary(show_warnings=False)])
        >>> # Disable device summary completely by not including the callback
        >>> trainer = Trainer(callbacks=[], enable_device_summary=False)

    """

    def __init__(self, show_warnings: bool = True) -> None:
        self._show_warnings = show_warnings
        self._logged = False

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Log device information at the start of any training stage.

        The device summary is only logged once per Trainer instance, even if setup is called multiple times (e.g., for
        fit then test).

        """
        if self._logged:
            return
        self._logged = True
        self._log_device_info(trainer)

    def _log_device_info(self, trainer: "pl.Trainer") -> None:
        """Log information about available and used devices."""
        if CUDAAccelerator.is_available():
            gpu_available = True
            gpu_type = " (cuda)"
        elif MPSAccelerator.is_available():
            gpu_available = True
            gpu_type = " (mps)"
        else:
            gpu_available = False
            gpu_type = ""

        gpu_used = isinstance(trainer.accelerator, (CUDAAccelerator, MPSAccelerator))
        rank_zero_info(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}")

        num_tpu_cores = trainer.num_devices if isinstance(trainer.accelerator, XLAAccelerator) else 0
        rank_zero_info(f"TPU available: {XLAAccelerator.is_available()}, using: {num_tpu_cores} TPU cores")

        if not self._show_warnings:
            return

        if (
            CUDAAccelerator.is_available()
            and not isinstance(trainer.accelerator, CUDAAccelerator)
            or MPSAccelerator.is_available()
            and not isinstance(trainer.accelerator, MPSAccelerator)
        ):
            rank_zero_warn(
                "GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.",
                category=PossibleUserWarning,
            )

        if XLAAccelerator.is_available() and not isinstance(trainer.accelerator, XLAAccelerator):
            rank_zero_warn("TPU available but not used. You can set it by doing `Trainer(accelerator='tpu')`.")
