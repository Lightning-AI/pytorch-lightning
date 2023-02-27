# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import time

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="~/data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

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

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    fabric = L.Fabric(accelerator="cpu")

    setup_t0 = time.perf_counter()
    net, optimizer = fabric.setup(net, optimizer)
    trainloader = fabric.setup_dataloaders(trainloader)
    setup_t1 = time.perf_counter()
    print(f"setup time: {setup_t1-setup_t0} sec")

    iteration_timings = []
    iterator = iter(trainloader)
    while True:
        t0 = time.perf_counter()
        try:
            data = next(iterator)
        except StopIteration:
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    """
    median tensor(0.0014)
    mean tensor(0.0020)
    std tensor(0.0502)
    """
    print("median", torch.median(torch.tensor(iteration_timings)))
    print("mean", torch.mean(torch.tensor(iteration_timings)))
    print("std", torch.std(torch.tensor(iteration_timings)))


if __name__ == "__main__":
    main()
