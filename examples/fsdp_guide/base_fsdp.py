from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2

policy = partial(
    transformer_auto_wrap_policy, transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
)

strategy = FSDPStrategy(
    auto_wrap_policy=policy,
    # activation_checkpointing=[
    #     nn.TransformerEncoderLayer,
    #     nn.TransformerDecoderLayer,
    # ],
    # cpu_offload=True,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)

# Regular
# allocated 11547, active 12780, reserved 15066

# Activation checkpointing
# allocated 11579, active 11602, reserved 15716

fabric = L.Fabric(devices=2, strategy=strategy)
fabric.launch()

with fabric.rank_zero_first():
    dataset = WikiText2()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

with fabric.init_module(empty_init=False):
    model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)

optimizer = torch.optim.Adam(model.parameters())

model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)

import time

timings = []

for i, (input, target) in enumerate(dataloader):
    if i >= 100:
        break

    t0 = time.time()

    output = model(input, target)
    loss = F.nll_loss(output, target.view(-1))
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(loss.item())

    timings.append(time.time() - t0)

latest = timings[10:]
print(f"{sum(latest) / len(latest):.2f}")

fabric.print(torch.cuda.memory_summary())
