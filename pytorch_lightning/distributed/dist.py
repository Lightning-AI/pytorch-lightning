import torch
from typing import Any
import pickle
import numpy as np
import torch.distributed as torch_distrib


def broadcast(self, x: Any, rank: int):
    is_tensor = isinstance(x, torch.Tensor)
    if not is_tensor:
        x = self._encode(x)
    torch_distrib.broadcast(x, src=rank)
    
    print('-' * 100)
    print(x)
    print(rank)
    print('-' * 100)

    if not is_tensor:
        x = _decode(x)
    return x


def _encode(obj):
    data = pickle.dumps(obj)
    data = np.frombuffer(data, dtype=np.uint8)
    return torch.from_numpy(data)


def _decode(tensor):
    data = tensor.numpy().tobytes()
    return pickle.loads(data)
