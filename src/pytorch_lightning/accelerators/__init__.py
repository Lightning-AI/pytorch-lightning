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
from lightning_fabric.accelerators import find_usable_musa_devices  # noqa: F401
from lightning_fabric.accelerators.registry import _AcceleratorRegistry, call_register_accelerators
from pytorch_lightning.accelerators.accelerator import Accelerator  # noqa: F401
from pytorch_lightning.accelerators.cpu import CPUAccelerator  # noqa: F401
from pytorch_lightning.accelerators.musa import MUSAAccelerator  # noqa: F401
from pytorch_lightning.accelerators.hpu import HPUAccelerator  # noqa: F401
from pytorch_lightning.accelerators.ipu import IPUAccelerator  # noqa: F401
from pytorch_lightning.accelerators.mps import MPSAccelerator  # noqa: F401
from pytorch_lightning.accelerators.tpu import TPUAccelerator  # noqa: F401

ACCELERATORS_BASE_MODULE = "pytorch_lightning.accelerators"
AcceleratorRegistry = _AcceleratorRegistry()
call_register_accelerators(AcceleratorRegistry, ACCELERATORS_BASE_MODULE)
