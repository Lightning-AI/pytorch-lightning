import time

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataloader(dataset_size=100, batch_size=4):
    inputs = torch.rand(dataset_size, 3, 32, 32)
    labels = torch.randint(0, 10, (dataset_size, ))
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return dataloader


def make_deterministic():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def train_torch(steps=100, batch_size=4):
    make_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    dataloader = get_dataloader(dataset_size=(steps * batch_size), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return dict(iteration_timings=torch.tensor(iteration_timings))


def train_fabric(steps=100, batch_size=4):
    make_deterministic()
    fabric = L.Fabric(accelerator="cpu")

    net = Net()
    dataloader = get_dataloader(dataset_size=(steps * batch_size), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net, optimizer = fabric.setup(net, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return dict(iteration_timings=torch.tensor(iteration_timings))


def compare():
    outputs_torch = train_torch(steps=2000)
    outputs_fabric = train_fabric(steps=2000)

    # 3.5009579733014107e-06
    # 3.5009579733014107e-06
    median = torch.median(outputs_fabric["iteration_timings"]) - torch.median(outputs_torch["iteration_timings"])
    mean = torch.mean(outputs_fabric["iteration_timings"]) - torch.mean(outputs_torch["iteration_timings"])
    print("median", median.abs().item())
    print("mean", mean.abs().item())


if __name__ == "__main__":
    compare()
