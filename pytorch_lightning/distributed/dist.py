import torch
from typing import Any
import pickle
import numpy as np
import torch.distributed as torch_distrib


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, x: Any):
        is_tensor = isinstance(x, torch.Tensor)
        if not is_tensor:
            x = _encode(x).to(self.device)

        if self.rank > 0:
            x = torch.rand(1000).to(self.device)
        print('-' * 100)
        print(x)
        print(self.rank)
        print('-' * 100)
        torch_distrib.broadcast(x, src=self.rank)

        print('-' * 100)
        print(x)
        print(self.rank)
        print('-' * 100)

        if not is_tensor:
            x = _decode(x)
        return x


def _encode(obj):
    target = torch.zeros(1000) * -1
    data = pickle.dumps(obj)
    data = np.frombuffer(data, dtype=np.uint8)
    target[:len(data)] = data
    return torch.from_numpy(data)


def _decode(tensor):
    data = tensor.cpu().numpy().tobytes()
    torch_distrib.barrier()
    data = pickle.loads(data)
    return data
