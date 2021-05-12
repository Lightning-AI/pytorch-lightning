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
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.utilities import DeviceType, DistributedType, rank_zero_deprecation


class DeprecatedDistDeviceAttributes:

    num_gpus: int
    accelerator_connector: AcceleratorConnector

    @property
    def on_cpu(self) -> bool:
        rank_zero_deprecation("Internal: `on_cpu` is deprecated in v1.2 and will be removed in v1.4.")
        return self.accelerator_connector._device_type == DeviceType.CPU

    @on_cpu.setter
    def on_cpu(self, val: bool) -> None:
        rank_zero_deprecation("Internal: `on_cpu` is deprecated in v1.2 and will be removed in v1.4.")
        if val:
            self.accelerator_connector._device_type = DeviceType.CPU

    @property
    def on_tpu(self) -> bool:
        rank_zero_deprecation("Internal: `on_tpu` is deprecated in v1.2 and will be removed in v1.4.")
        return self.accelerator_connector._device_type == DeviceType.TPU

    @on_tpu.setter
    def on_tpu(self, val: bool) -> None:
        rank_zero_deprecation("Internal: `on_tpu` is deprecated in v1.2 and will be removed in v1.4.")
        if val:
            self.accelerator_connector._device_type = DeviceType.TPU

    @property
    def use_tpu(self) -> bool:
        rank_zero_deprecation("Internal: `use_tpu` is deprecated in v1.2 and will be removed in v1.4.")
        return self.on_tpu

    @use_tpu.setter
    def use_tpu(self, val: bool) -> None:
        rank_zero_deprecation("Internal: `use_tpu` is deprecated in v1.2 and will be removed in v1.4.")
        self.on_tpu = val

    @property
    def on_gpu(self) -> bool:
        rank_zero_deprecation("Internal: `on_gpu` is deprecated in v1.2 and will be removed in v1.4.")
        return self.accelerator_connector._device_type == DeviceType.GPU

    @on_gpu.setter
    def on_gpu(self, val: bool) -> None:
        rank_zero_deprecation("Internal: `on_gpu` is deprecated in v1.2 and will be removed in v1.4.")
        if val:
            self.accelerator_connector._device_type = DeviceType.GPU


class DeprecatedTrainerAttributes:

    accelerator: Accelerator
    lightning_module: LightningModule
    sanity_checking: bool

    @property
    def accelerator_backend(self) -> Accelerator:
        rank_zero_deprecation(
            "The `Trainer.accelerator_backend` attribute is deprecated in favor of `Trainer.accelerator`"
            " since 1.2 and will be removed in v1.4."
        )
        return self.accelerator

    def get_model(self) -> LightningModule:
        rank_zero_deprecation(
            "The use of `Trainer.get_model()` is deprecated in favor of `Trainer.lightning_module`"
            " and will be removed in v1.4."
        )
        return self.lightning_module

    @property
    def running_sanity_check(self) -> bool:
        rank_zero_deprecation(
            "`Trainer.running_sanity_check` has been renamed to `Trainer.sanity_checking` and will be removed in v1.5."
        )
        return self.sanity_checking
