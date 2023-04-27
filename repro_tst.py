import torch

model = torch.nn.Linear(3 * 32 * 32, 4)

for _ in range(1000):
    inputs = torch.rand(4, 3 * 32 * 32)
    loss = model(inputs).sum()
    print(loss.item())
    loss.backward()
