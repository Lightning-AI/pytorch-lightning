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
        torch_distrib.barrier()
        if self.rank == 0:
            # Emit data
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            data = bytearray(buffer.getbuffer())
            length_tensor = torch.tensor([len(data)]).long().to(self.device)
            length_tensor = torch_distrib.broadcast(length_tensor, src=0)
            data_tensor = torch.ByteTensor(data).to(self.device)
            data_tensor = torch_distrib.broadcast(data_tensor, src=0)
        else:
            # Fetch from the source
            length_tensor = torch.tensor([0]).long().to(self.device)
            length_tensor = torch_distrib.broadcast(length_tensor, src=0)
            data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
            data_tensor = torch_distrib.broadcast(data_tensor, src=0)
            buffer = io.BytesIO(data_tensor.numpy())
            obj = torch.load(buffer)

            print(obj)
        return obj

    def _encode(self, obj):
        padding = torch.zeros(100, device=self.device).long()
        result = [str(ord(c)) for c in obj]
        chunks = [torch.tensor(int(x)) for x in result]
        chunks = torch.stack(chunks).long()
        padding[:len(chunks)] = chunks
        padding = padding.to(self.device)
        return padding

    def _decode(self, tensor):
        tensor = tensor.long()
        print(tensor)
        chunks = [chr(x.cpu().numpy()) for x in tensor]
        text = ''.join(chunks)
        print(text, '|')
        return text
