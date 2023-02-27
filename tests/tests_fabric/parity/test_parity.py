# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import lightning as L
import torch
import torch.nn as nn
from tests_fabric.parity.utils import make_deterministic
from tests_fabric.parity.models import ConvNet


def train_torch(steps=100, batch_size=4):
    make_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNet().to(device)
    dataloader = model.get_dataloader(dataset_size=(steps * batch_size), batch_size=batch_size)
    loss_fn = model.get_loss_function()
    optimizer = model.get_optimizer()

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return dict(iteration_timings=torch.tensor(iteration_timings))


def train_fabric(steps=100, batch_size=4):
    make_deterministic()
    fabric = L.Fabric(accelerator="cpu")

    model = ConvNet()
    dataloader = model.get_dataloader(dataset_size=(steps * batch_size), batch_size=batch_size)
    loss_fn = model.get_loss_function()
    optimizer = model.get_optimizer()

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return dict(iteration_timings=torch.tensor(iteration_timings))


def test_compare():
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
