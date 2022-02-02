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
import os
import time
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Optional

import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers.ddp_spawn import _FakeQueue, _SpawnOutput, DDPSpawnLauncher
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_debug
from pytorch_lightning.utilities.model_helpers import is_overridden

if _TPU_AVAILABLE:
    import torch_xla.distributed.xla_multiprocessing as xmp
else:
    xm, xmp, MpDeviceLoader, rendezvous = [None] * 4


class TPUSpawnLauncher(DDPSpawnLauncher):
    def launch(self, trainer, function, *args, **kwargs):
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

    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_SpawnOutput"]:
        rank_zero_debug("Finalizing the TPU spawn environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = self.strategy.lightning_module.state_dict()

        # save the last weights
        weights_path = None
        if trainer.state.fn == TrainerFn.FITTING:
            weights_path = os.path.join(trainer.default_root_dir, ".temp.ckpt")
            self.strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)

        # We use `local_rank` here as separate filesystems are used for each VM for TPU Pod Training
        if self.strategy.local_rank != 0:
            return

        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        if is_overridden("add_to_queue", self.strategy.lightning_module):
            # TODO: Remove the if in v1.7
            self.strategy.lightning_module.add_to_queue(extra)
        self.strategy.add_to_queue(trainer, extra)

        return _SpawnOutput(best_model_path, weights_path, trainer.state, results, extra)
