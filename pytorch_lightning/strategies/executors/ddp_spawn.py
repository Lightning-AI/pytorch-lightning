import os
from collections import UserList
from typing import Any, NamedTuple, Optional

import torch
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.strategies.executors.base import Executor
from pytorch_lightning.trainer.states import TrainerFn, TrainerState
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.distributed import rank_zero_debug
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import _PATH


class DDPSpawnExecutor(Executor):
    def __init__(self, strategy):
        super().__init__(strategy)

    def execute(self, trainer, function, *args, **kwargs):
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
        if self.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

    def _recover_results_in_main_process(self, spawn_output: "_SpawnOutput", trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback:
            trainer.checkpoint_callback.best_model_path = spawn_output.best_model_path

        # TODO: pass also best score
        # load last weights
        if spawn_output.weights_path is not None:
            ckpt = self.checkpoint_io.load_checkpoint(
                spawn_output.weights_path, map_location=(lambda storage, loc: storage)
            )
            self.lightning_module.load_state_dict(ckpt)
            self.checkpoint_io.remove_checkpoint(spawn_output.weights_path)

        trainer.state = spawn_output.trainer_state

        # get the `callback_metrics` and set it to the trainer
        if is_overridden("get_from_queue", self.lightning_module):
            # only in case the user does not override it.
            # TODO: Remove the if in v1.7
            self.lightning_module.get_from_queue(spawn_output.extra)
        self.get_from_queue(trainer, spawn_output.extra)

    def _collect_ranku_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_SpawnOutput"]:
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
        if is_overridden("add_to_queue", self.lightning_module):
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
