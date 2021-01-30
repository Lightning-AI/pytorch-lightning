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
from pytorch_lightning.accelerators.legacy.accelerator import Accelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.cpu_accelerator import CPUAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp2_accelerator import DDP2Accelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp_accelerator import DDPAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp_cpu_hpc_accelerator import DDPCPUHPCAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp_cpu_spawn_accelerator import DDPCPUSpawnAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp_hpc_accelerator import DDPHPCAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.ddp_spawn_accelerator import DDPSpawnAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.dp_accelerator import DataParallelAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.gpu_accelerator import GPUAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.horovod_accelerator import HorovodAccelerator  # noqa: F401
from pytorch_lightning.accelerators.legacy.tpu_accelerator import TPUAccelerator  # noqa: F401
