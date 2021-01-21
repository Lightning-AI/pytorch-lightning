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
from pytorch_lightning.accelerators.cpu_accelerator import CPUAccelerator
from pytorch_lightning.accelerators.ddp2_accelerator import DDP2Accelerator
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
from pytorch_lightning.accelerators.ddp_spawn_accelerator import DDPSpawnAccelerator
from pytorch_lightning.accelerators.ddp_cpu_spawn_accelerator import DDPCPUSpawnAccelerator
from pytorch_lightning.accelerators.dp_accelerator import DataParallelAccelerator
from pytorch_lightning.accelerators.gpu_accelerator import GPUAccelerator
from pytorch_lightning.accelerators.tpu_accelerator import TPUAccelerator
from pytorch_lightning.accelerators.horovod_accelerator import HorovodAccelerator
from pytorch_lightning.accelerators.ddp_hpc_accelerator import DDPHPCAccelerator
from pytorch_lightning.accelerators.ddp_cpu_hpc_accelerator import DDPCPUHPCAccelerator
from pytorch_lightning.accelerators.accelerator import Accelerator
