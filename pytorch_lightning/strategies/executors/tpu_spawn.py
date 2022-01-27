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
import time
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable

import torch.multiprocessing as mp

from pytorch_lightning.strategies.executors.ddp_spawn import DDPSpawnExecutor
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device

if _TPU_AVAILABLE:
    import torch_xla.distributed.xla_multiprocessing as xmp
else:
    xm, xmp, MpDeviceLoader, rendezvous = [None] * 4


class TPUSpawnExecutor(DDPSpawnExecutor):
    def execute(self, trainer, function, *args, **kwargs):
        context = mp.get_context(self.start_method or "fork")
        return_queue = context.SimpleQueue()
        xmp.spawn(self._wrapped_function, args=(function, args, kwargs, return_queue), **self.get_mp_spawn_kwargs())
        return return_queue.get()

    def _wrapped_function(
        self, process_idx: int, function: Callable, args: Any, kwargs: Any, return_queue: SimpleQueue
    ) -> None:
        self._worker_setup(process_idx)
        result = function(*args, **kwargs)
        if self.local_rank == 0:
            return_queue.put(move_data_to_device(result, "cpu"))

        # https://github.com/pytorch/xla/issues/1801#issuecomment-602799542
        self.barrier("end-process")

        # Ensure that the rank 0 process is the one exiting last
        # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
        if self.local_rank == 0:
            time.sleep(2)
