from abc import ABC, abstractmethod

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.inference.functional import infer as func_infer


class InferenceABC(torch.nn.Module, ABC):
    def __init__(self, network: LightningModule, infer_mode='eval'):
        super().__init__()
        self.network = network
        self.infer_mode = infer_mode
        self.infer = lambda x: func_infer.infer(self.network, x, self.infer_mode)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Inference(InferenceABC):
    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)

    def forward(self, x):
        return self.infer(x)


class TempScalingInference(InferenceABC):
    def __init__(self, network, T:float = 1.5, **kwargs):
        super().__init__(network, **kwargs)
        self.T = T
        if network.val_dataloader():
            self._temperature_set(self.network.val_dataloader())

    def forward(self, x, T=None):
        T = T if not T else self.T
        return self.infer(x) / T

    def _temperature_set(self, val_dataloader, lr=1e-2, max_iter=50):
        # from https://github.com/gpleiss/temperature_scaling/
        preds = []
        targets = [] 
        T = torch.nn.Parameter(torch.ones(1) * self.T)
        with torch.no_grad():
            for x, y in val_dataloader:
                preds.append(self(x))
                targets.append(y)

        optim = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
        def opt():
            loss = torch.nn.CrossEntropyLoss()
            loss.backward()
            return loss
        optim.step(opt)

        self.T = float(T)


class IterativeInference(InferenceABC):
    def __init__(self, network, N:int = 1, infer_mode = 'train', return_mode = 'avg', **kwargs):
        super().__init__(network, infer_mode, **kwargs)
        RETURN_MODE = {
            'avg': lambda x: torch.mean(torch.stack(x), 0),
            'list': lambda x: x
        }
 
        self.N = 1
        self.return_mode = RETURN_MODE[return_mode]

    def forward(self, x, N=None):
        self.infer_mode()
        N = N if not N else self.N

        return self.return_mode([self.infer(x) for _ in range(N)])
