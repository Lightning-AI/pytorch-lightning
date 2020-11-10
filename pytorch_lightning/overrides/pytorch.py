import torch.distributed as dist
from torch.cuda.amp import GradScaler


class ShardedGradScaler(GradScaler):
    def step(self, optimizer, *args, **kwargs):
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        for v in optimizer_state["found_inf_per_device"].values():
            dist.all_reduce(v)

        return super().step(optimizer, *args, **kwargs)
