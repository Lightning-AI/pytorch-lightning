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

from pytorch_lightning.utilities import rank_zero_warn, DistributedType, DeviceType


class DeprecatedDistDeviceAttributes:

    _distrib_type: DistributedType
    _device_type: DeviceType
    num_gpus: int

    @property
    def on_cpu(self) -> bool:
        # rank_zero_warn("Internal: `on_cpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.CPU

    @on_cpu.setter
    def on_cpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_cpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._device_type = DeviceType.CPU

    @property
    def on_tpu(self) -> bool:
        # rank_zero_warn("Internal: `on_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.TPU

    @on_tpu.setter
    def on_tpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if TPU is missing
        if val:
            self._device_type = DeviceType.TPU

    @property
    def use_tpu(self) -> bool:
        # rank_zero_warn("Internal: `use_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.TPU

    @use_tpu.setter
    def use_tpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if TPU is missing
        if val:
            self._device_type = DeviceType.TPU

    @property
    def on_gpu(self) -> bool:
        # rank_zero_warn("Internal: `on_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.GPU

    @on_gpu.setter
    def on_gpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if GPU is missing
        if val:
            self._device_type = DeviceType.GPU

    @property
    def use_dp(self) -> bool:
        # rank_zero_warn("Internal: `use_dp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DP

    @use_dp.setter
    def use_dp(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_dp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DP

    @property
    def use_ddp(self) -> bool:
        # rank_zero_warn("Internal: `use_ddp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DDP

    @use_ddp.setter
    def use_ddp(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_ddp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DDP

    @property
    def use_ddp2(self) -> bool:
        # rank_zero_warn("Internal: `use_ddp2` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DDP2

    @use_ddp2.setter
    def use_ddp2(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_ddp2` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DDP2

    @property
    def use_horovod(self) -> bool:
        # rank_zero_warn(
        #     "Internal: `use_horovod` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning
        # )
        return self._device_type and self._distrib_type == DistributedType.HOROVOD

    @use_horovod.setter
    def use_horovod(self, val: bool) -> None:
        # rank_zero_warn(
        #     "Internal: `use_horovod` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning
        # )
        if val:
            self._distrib_type = DistributedType.HOROVOD

    @property
    def use_single_gpu(self) -> bool:
        # rank_zero_warn(
        #     "Internal: `use_single_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning,
        # )
        # todo, limiting to exclude DDP2 is not clear but it comes from connectors...
        return (self._device_type and self._device_type == DeviceType.GPU
                and self.num_gpus == 1
                and self._distrib_type not in (DistributedType.DDP2, ))

    @use_single_gpu.setter
    def use_single_gpu(self, val: bool) -> None:
        # rank_zero_warn(
        #     "Internal: `use_single_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning,
        # )
        if val:
            self._device_type = DeviceType.GPU
