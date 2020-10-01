import io
import torch
from typing import Any


class LightningDistributed:

    def __init__(self, trainer, rank=None, device=None):
        self.rank = rank
        self.device = device
        self.trainer = trainer

    def broadcast(self, obj: Any):
        if self.rank == 0:
            self._emit(obj)
        else:
            obj = self._receive()
        return obj

    def _emit(self, obj):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.tensor([len(data)]).long().to(self.device)
        length_tensor = self.trainer.accelerator_backend.broadcast(length_tensor, src=0)
        data_tensor = torch.ByteTensor(data).to(self.device)
        data_tensor = self.trainer.accelerator_backend.broadcast(data_tensor, src=0)

    def _receive(self):
        length_tensor = torch.tensor([0]).long().to(self.device)
        self.trainer.accelerator_backend.broadcast(length_tensor, src=0)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
        self.trainer.accelerator_backend.broadcast(data_tensor, src=0)
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer)
        return obj
