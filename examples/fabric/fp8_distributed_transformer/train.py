import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from tqdm import tqdm

import lightning as L
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.pytorch.demos import Transformer, WikiText2


def configure_model(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
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
        if isinstance(module, (torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer)):
            fully_shard(module, mesh=device_mesh)

    fully_shard(model, mesh=device_mesh)

    return torch.compile(model)


def train():
    L.seed_everything(42)

    batch_size = 8
    micro_batch_size = 1

    max_steps = 100

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=8, batch_size=micro_batch_size)

    with torch.device("meta"):
        model = Transformer(
            vocab_size=dataset.vocab_size,
            nlayers=16,
            nhid=4096,
            ninp=1024,
            nhead=32,
        )

    strategy = ModelParallelStrategy(data_parallel_size=4, tensor_parallel_size=1, parallelize_fn=configure_model)

    fabric = L.Fabric(precision="bf16-true", strategy=strategy)
    fabric.launch()

    model = fabric.setup(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = fabric.setup_optimizers(optimizer)

    dataloader = fabric.setup_dataloaders(dataloader)

    iterable = tqdm(enumerate(dataloader), total=len(dataloader)) if fabric.is_global_zero else enumerate(dataloader)

    steps = 0

    for i, batch in iterable:
        input, target = batch

        is_accumulating = i % (batch_size // micro_batch_size) != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            output = model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            fabric.backward(loss)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        if fabric.is_global_zero:
            iterable.set_postfix_str(f"train_loss={loss.item():.2f}")

        if steps == max_steps:
            break

    fabric.print(torch.cuda.memory_summary())


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train()
