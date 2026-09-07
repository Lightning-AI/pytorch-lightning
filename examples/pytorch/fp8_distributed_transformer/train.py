import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2
from lightning.pytorch.strategies import ModelParallelStrategy


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return

        with torch.device("meta"):
            model = Transformer(
                vocab_size=self.vocab_size,
                nlayers=16,
                nhid=4096,
                ninp=1024,
                nhead=32,
            )

        float8_config = Float8LinearConfig(
            # pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly  # noqa
            pad_inner_dim=True,
        )

        def module_filter_fn(mod: torch.nn.Module, fqn: str):
            # we skip the decoder because it typically vocabulary size
            # is not divisible by 16 as required by float8
            return fqn != "decoder"

        convert_to_float8_training(model, config=float8_config, module_filter_fn=module_filter_fn)

        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                fully_shard(module, mesh=self.device_mesh)

        fully_shard(model, mesh=self.device_mesh)

        self.model = torch.compile(model)

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def train():
    L.seed_everything(42)

    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, num_workers=8, batch_size=1)

    model = LanguageModel(vocab_size=dataset.vocab_size)

    mp_strategy = ModelParallelStrategy(
        data_parallel_size=4,
        tensor_parallel_size=1,
    )

    trainer = L.Trainer(strategy=mp_strategy, max_steps=100, precision="bf16-true", accumulate_grad_batches=8)

    trainer.fit(model, train_dataloader)

    trainer.print(torch.cuda.memory_summary())


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train()
