import atexit
import os
from typing import Optional

import torch
import torch.distributed

from pytorch_lightning.utilities import rank_zero_info


class TensorRunningAccum(object):
    """Tracks a running accumulation values (min, max, mean) without graph
    references.

    Examples:
        >>> accum = TensorRunningAccum(5)
        >>> accum.last(), accum.mean()
        (None, None)
        >>> accum.append(torch.tensor(1.5))
        >>> accum.last(), accum.mean()
        (tensor(1.5000), tensor(1.5000))
        >>> accum.append(torch.tensor(2.5))
        >>> accum.last(), accum.mean()
        (tensor(2.5000), tensor(2.))
        >>> accum.reset()
        >>> _= [accum.append(torch.tensor(i)) for i in range(13)]
        >>> accum.last(), accum.mean(), accum.min(), accum.max()
        (tensor(12.), tensor(10.), tensor(8.), tensor(12.))
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.memory = torch.Tensor(self.window_length)
        self.current_idx: int = 0
        self.last_idx: Optional[int] = None
        self.rotated: bool = False

    def reset(self) -> None:
        """Empty the accumulator."""
        self = TensorRunningAccum(self.window_length)

    def last(self):
        """Get the last added element."""
        if self.last_idx is not None:
            return self.memory[self.last_idx]

    def append(self, x):
        """Add an element to the accumulator."""
        # ensure same device and type
        if self.memory.device != x.device or self.memory.type() != x.type():
            x = x.to(self.memory)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x
            self.last_idx = self.current_idx

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        self.current_idx = self.current_idx % self.window_length
        if self.current_idx == 0:
            self.rotated = True

    def mean(self):
        """Get mean value from stored elements."""
        return self._agg_memory('mean')

    def max(self):
        """Get maximal value from stored elements."""
        return self._agg_memory('max')

    def min(self):
        """Get minimal value from stored elements."""
        return self._agg_memory('min')

    def _agg_memory(self, how: str):
        if self.last_idx is not None:
            if self.rotated:
                return getattr(self.memory, how)()
            else:
                return getattr(self.memory[:self.current_idx], how)()


class Accumulator(object):
    def __init__(self):
        self.num_values = 0
        self.total = 0

    def accumulate(self, x):
        with torch.no_grad():
            self.total += x
            self.num_values += 1

    def mean(self):
        return self.total / self.num_values


class DistributedConnection:

    def __init__(self, trainer):
        super().__init__()
        # self.world_size = world_size
        # self.is_slurm_managing_tasks = is_slurm_managing_tasks
        self.trainer = trainer
        # self._is_initialized = False
        #if self.trainer.gl

        # initial random port, before ddp connection is initialized
        self.trainer.set_random_port()

    def init_connection(self, trainer, model):
        if torch.distributed.is_initialized():
            print("ddp connection already initialized, moving to new port")

            torch.distributed.barrier()

            if trainer.global_rank == 0:
                print('sending new port to others')
                new_port = trainer.set_random_port(force=True, overwrite=False)
                torch.distributed.broadcast(torch.tensor(new_port, device=model.device), src=0)
            else:
                print('receiving new port on rank=', trainer.global_rank)
                new_port = torch.empty(1, device=model.device)
                torch.distributed.broadcast(new_port, trainer.global_rank)
                new_port = int(new_port.item())

            torch.distributed.destroy_process_group()
            os.environ['MASTER_PORT'] = str(new_port)

        model.init_ddp_connection(trainer.global_rank, trainer.world_size, trainer.is_slurm_managing_tasks)

        def exit_handler():
            if torch.distributed.is_initialized():
                # torch.distributed.barrier()
                torch.distributed.destroy_process_group()

            print('group destroyed on ', trainer.global_rank)

        # atexit.register(exit_handler)

    def teardown(self):
        return
        if torch.distributed.is_initialized():
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
