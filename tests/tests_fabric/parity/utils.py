from contextlib import contextmanager
from typing import Generator

import torch
from torch import nn


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.0001)


@contextmanager
def precision_context(precision, accelerator) -> Generator[None, None, None]:
    if precision == 32:
        yield
        return
    if accelerator == "gpu":
        with torch.cuda.amp.autocast():
            yield
    elif accelerator == "cpu":
        with torch.cpu.amp.autocast():
            yield
