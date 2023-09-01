from typing import Tuple

import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning.pytorch.demos.transformer import Transformer


class LightningTransformer(pl.LightningModule):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        input, target = batch
        return self.model(input.view(1, -1), target.view(1, -1))

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input, target = batch
        output = self.model(input.view(1, -1), target.view(1, -1))
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def predict_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        return self(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
