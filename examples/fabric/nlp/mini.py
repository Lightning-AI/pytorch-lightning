import torch
import torch.nn.functional as F
from lightning.pytorch.demos import WikiText2, Transformer

import lightning as L


fabric = L.Fabric()

dataset = WikiText2()
dataloader = torch.utils.data.DataLoader(dataset)
model = Transformer(vocab_size=dataset.vocab_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
for epoch in range(20):
    for batch in dataloader:
        input, target = batch
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        optimizer.step()
