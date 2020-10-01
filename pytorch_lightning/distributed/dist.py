import torch
from typing import Any
import pickle
import numpy as np
import torch.distributed as torch_distrib
import io


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any):
        if self.rank == 0:
            print(f'sending from: {self.rank}')
            self._emit(obj)
        else:
            obj = self._receive()
            print(f'receiving at: {self.rank}')
            print('obj')
        return obj

    def _emit(self, obj):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.tensor([len(data)]).long().to(self.device)
        length_tensor = torch_distrib.broadcast(length_tensor, src=0)
        data_tensor = torch.ByteTensor(data).to(self.device)
        data_tensor = torch_distrib.broadcast(data_tensor, src=0)

    def _receive(self):
        length_tensor = torch.tensor([0]).long().to(self.device)
        torch_distrib.broadcast(length_tensor, src=0)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
        torch_distrib.broadcast(data_tensor, src=0)
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer)
        return obj
