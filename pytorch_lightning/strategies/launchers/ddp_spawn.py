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
from collections import UserList
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers.base import Launcher
from pytorch_lightning.trainer.states import TrainerFn, TrainerState
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_debug
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import _PATH


class DDPSpawnLauncher(Launcher):
    def launch(self, trainer, function, *args, **kwargs):
        os.environ["MASTER_PORT"] = str(self.strategy.cluster_environment.main_port)
        context = mp.get_context("spawn")
        return_queue = context.SimpleQueue()
        mp.spawn(
            self._wrapped_function,
            args=(trainer, function, args, kwargs, return_queue),
            nprocs=self.strategy.num_processes,
        )
        spawn_output = return_queue.get()
        self._recover_results_in_main_process(spawn_output, trainer)
        return spawn_output.trainer_results

    def _wrapped_function(self, process_idx, trainer, function, args, kwargs, return_queue):
        self.strategy._worker_setup(process_idx)
        results = function(*args, **kwargs)
        results = self._collect_rank_zero_results(trainer, results)
        if self.strategy.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

    def _recover_results_in_main_process(self, spawn_output: "_SpawnOutput", trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = spawn_output.best_model_path

        # TODO: pass also best score
        # load last weights
        if spawn_output.weights_path is not None:
            ckpt = self.strategy.checkpoint_io.load_checkpoint(
                spawn_output.weights_path, map_location=(lambda storage, loc: storage)
            )
            self.strategy.lightning_module.load_state_dict(ckpt)
            self.strategy.checkpoint_io.remove_checkpoint(spawn_output.weights_path)

        trainer.state = spawn_output.trainer_state

        # get the `callback_metrics` and set it to the trainer
        if is_overridden("get_from_queue", self.strategy.lightning_module):
            # only in case the user does not override it.
            # TODO: Remove the if in v1.7
            self.strategy.lightning_module.get_from_queue(spawn_output.extra)
        self.get_from_queue(trainer, spawn_output.extra)

    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_SpawnOutput"]:
        rank_zero_debug("Finalizing the DDP spawn environment.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = self.strategy.lightning_module.state_dict()

        if self.strategy.global_rank != 0:
            return

        # save the last weights
        weights_path = None
        if trainer.state.fn == TrainerFn.FITTING:
            weights_path = os.path.join(trainer.default_root_dir, ".temp.ckpt")
            self.strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)

        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        if is_overridden("add_to_queue", self.strategy.lightning_module):
            # TODO: Remove the if in v1.7
            self.strategy.lightning_module.add_to_queue(extra)
        self.add_to_queue(trainer, extra)

        return _SpawnOutput(best_model_path, weights_path, trainer.state, results, extra)

    def add_to_queue(self, trainer: "pl.Trainer", queue: "_FakeQueue") -> None:
        """Appends the :attr:`trainer.callback_metrics` dictionary to the given queue. To avoid issues with memory
        sharing, we cast the data to numpy.

        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue to append the data.
        """
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, torch.Tensor, lambda x: x.cpu().numpy()
        )  # send as numpy to avoid issues with memory sharing
        queue.put(callback_metrics)

    def get_from_queue(self, trainer: "pl.Trainer", queue: "_FakeQueue") -> None:
        """Retrieve the :attr:`trainer.callback_metrics` dictionary from the given queue. To preserve consistency,
        we cast back the data to ``torch.Tensor``.

        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue from where to get the data.
        """
        # NOTE: `add_to_queue` needs to be called before
        callback_metrics: dict = queue.get()
        trainer.callback_metrics.update(apply_to_collection(callback_metrics, np.ndarray, lambda x: torch.tensor(x)))


class _FakeQueue(UserList):
    """Simulates a :class:`torch.multiprocessing.queue.SimpleQueue` interface using the Python list."""

    def get(self) -> Any:
        return self.pop(0)

    def put(self, item: Any) -> None:
        self.append(item)

    def empty(self) -> bool:
        return len(self) == 0


class _SpawnOutput(NamedTuple):
    best_model_path: Optional[_PATH]
    weights_path: Optional[_PATH]
    trainer_state: TrainerState
    trainer_results: Any
    extra: _FakeQueue
