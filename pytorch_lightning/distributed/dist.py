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
            x = self._encode(x).to(self.device)

        if self.rank > 0:
            x = torch.rand(100).to(self.device).long()
        torch_distrib.broadcast(x, src=self.rank)

        if not is_tensor:
            x = self._decode(x)
        return x

    def _encode(self, obj):
        padding = torch.zeros(100, device=self.device).long()
        result = [str(ord(c)) for c in obj]
        chunks = [torch.tensor(int(x)) for x in result]
        chunks = torch.stack(chunks).long()
        padding[:len(chunks)] = chunks
        return padding

    def _decode(self, tensor):
        tensor = tensor.long()
        print(tensor)
        chunks = [chr(x.cpu().numpy()) for x in tensor]
        text = ''.join(chunks)
        print(text, '|')
        return text
