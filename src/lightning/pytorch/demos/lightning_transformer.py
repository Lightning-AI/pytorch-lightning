import pytorch_lightning as pl
import torch

from pytorch_lightning.demos.transformer import Transformer


class LightningTransformer(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, batch):
        input, target = batch
        return self.model(input.view(1, -1), target.view(1, -1))

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input.view(1, -1), target.view(1, -1))
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def predict_step(self, batch):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
