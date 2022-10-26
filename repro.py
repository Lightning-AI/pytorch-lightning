import torch
import torch.nn as nn

from torch.optim import Adam

model = nn.Linear(2, 2)
optimizer = Adam(model.parameters())

print(optimizer.param_groups[0]["params"])