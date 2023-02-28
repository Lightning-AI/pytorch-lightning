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

from lightning.fabric import Fabric
import torch
from tests_fabric.parity.utils import make_deterministic
from tests_fabric.parity.models import ConvNet


def train_torch(rank=0, accelerator="cpu", devices=1, num_steps=100, batch_size=4):
    make_deterministic()
    device = torch.device("cuda" if accelerator == "cuda" else "cpu", rank)
    model = ConvNet().to(device)
    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    loss_fn = model.get_loss_function()
    optimizer = model.get_optimizer()

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(num_steps):
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

    return torch.tensor(iteration_timings)


def train_fabric(num_steps=100, batch_size=4):
    make_deterministic()
    fabric = Fabric(accelerator="cpu")
    fabric.launch()

    model = ConvNet()
    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    loss_fn = model.get_loss_function()
    optimizer = model.get_optimizer()

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(num_steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return torch.tensor(iteration_timings)


def launch_fabric():
    fabric = Fabric()
    fabric.launch(train_fabric, **kwargs)


def test_parity_cpu():
    timings_torch = train_torch(num_steps=2000)
    timings_fabric = train_fabric(num_steps=2000)

    # The median is more robust to outliers than the mean
    assert torch.isclose(torch.median(timings_torch), torch.median(timings_fabric), rtol=1e-4, atol=1e-4)
