# Copyright The Lightning AI team.
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
from lightning.fabric.accelerators.accelerator import Accelerator  # noqa: F401
from lightning.fabric.accelerators.cpu import CPUAccelerator  # noqa: F401
from lightning.fabric.accelerators.cuda import CUDAAccelerator, find_usable_cuda_devices  # noqa: F401
from lightning.fabric.accelerators.mps import MPSAccelerator  # noqa: F401
from lightning.fabric.accelerators.registry import _AcceleratorRegistry, call_register_accelerators
from lightning.fabric.accelerators.tpu import TPUAccelerator  # noqa: F401

_ACCELERATORS_BASE_MODULE = "lightning.fabric.accelerators"
ACCELERATOR_REGISTRY = _AcceleratorRegistry()
call_register_accelerators(ACCELERATOR_REGISTRY, _ACCELERATORS_BASE_MODULE)
