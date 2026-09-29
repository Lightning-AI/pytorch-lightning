<div align="center">

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric_logo.png" width="400px">

**Fabric is the fast and lightweight way to scale PyTorch models without boilerplate**

______________________________________________________________________

<p align="center">
  <a href="https://lightning.ai/">Website</a> •
  <a href="https://lightning.ai/docs/fabric/">Docs</a> •
  <a href="#getting-started">Getting started</a> •
  <a href="#faq">FAQ</a> •
  <a href="#asking-for-help">Help</a> •
  <a href="https://discord.gg/VptPCZkGNa">Discord</a>
</p>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning_fabric)](https://pypi.org/project/lightning_fabric/)
[![PyPI Status](https://badge.fury.io/py/lightning_fabric.svg)](https://badge.fury.io/py/lightning_fabric)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lightning-fabric)](https://pepy.tech/project/lightning-fabric)
[![Conda](https://img.shields.io/conda/v/conda-forge/lightning_fabric?label=conda&color=success)](https://anaconda.org/conda-forge/lightning_fabric)

</div>

# Lightning Fabric: Expert control.

Run on any device at any scale with expert-level control over PyTorch training loop and scaling strategy. You can even write your own Trainer.

Fabric is designed for the most complex models like foundation model scaling, LLMs, diffusion, transformers, reinforcement learning, active learning. Of any size.

<table>
<tr>
<th>What to change</th>
<th>Resulting Fabric Code (copy me!)</th>
</tr>
<tr>
<td>
<sub>

```diff
+ import lightning as L
  import torch; import torchvision as tv

  dataset = tv.datasets.CIFAR10("data", download=True,
                                train=True,
                                transform=tv.transforms.ToTensor())

+ fabric = L.Fabric()
+ fabric.launch()

  model = tv.models.resnet18()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
- device = "cuda" if torch.cuda.is_available() else "cpu"
- model.to(device)
+ model, optimizer = fabric.setup(model, optimizer)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
+ dataloader = fabric.setup_dataloaders(dataloader)

  model.train()
  num_epochs = 10
  for epoch in range(num_epochs):
      for batch in dataloader:
          inputs, labels = batch
-         inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = torch.nn.functional.cross_entropy(outputs, labels)
-         loss.backward()
+         fabric.backward(loss)
          optimizer.step()
```

</sub>
<td>
<sub>

```Python
import lightning as L
import torch; import torchvision as tv

dataset = tv.datasets.CIFAR10("data", download=True,
                              train=True,
                              transform=tv.transforms.ToTensor())

fabric = L.Fabric()
fabric.launch()

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model, optimizer = fabric.setup(model, optimizer)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
```

</sub>
</td>
</tr>
</table>

## Key features

<details>
  <summary>Easily switch from running on CPU to GPU (Apple Silicon, CUDA, …), TPU, multi-GPU or even multi-node training</summary>

```python
# Use your available hardware
# no code changes needed
fabric = Fabric()

# Run on GPUs (CUDA or MPS)
fabric = Fabric(accelerator="gpu")

# 8 GPUs
fabric = Fabric(accelerator="gpu", devices=8)

# 256 GPUs, multi-node
fabric = Fabric(accelerator="gpu", devices=8, num_nodes=32)

# Run on TPUs
fabric = Fabric(accelerator="tpu")
```

</details>

<details>
  <summary>Use state-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed) and mixed precision out of the box</summary>

```python
# Use state-of-the-art distributed training techniques
fabric = Fabric(strategy="ddp")
fabric = Fabric(strategy="deepspeed")
fabric = Fabric(strategy="fsdp")

# Switch the precision
fabric = Fabric(precision="16-mixed")
fabric = Fabric(precision="64")
```

</details>

<details>
  <summary>All the device logic boilerplate is handled for you</summary>

```diff
  # no more of this!
- model.to(device)
- batch.to(device)
```

</details>

<details>
  <summary>Build your own custom Trainer using Fabric primitives for training checkpointing, logging, and more</summary>

```python
import lightning as L


class MyCustomTrainer:
    def __init__(self, accelerator="auto", strategy="auto", devices="auto", precision="32-true"):
        self.fabric = L.Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision)

    def fit(self, model, optimizer, dataloader, max_epochs):
        self.fabric.launch()

        model, optimizer = self.fabric.setup(model, optimizer)
        dataloader = self.fabric.setup_dataloaders(dataloader)
        model.train()

        for epoch in range(max_epochs):
            for batch in dataloader:
                input, target = batch
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                self.fabric.backward(loss)
                optimizer.step()
```

You can find a more extensive example in our [examples](../../examples/fabric/build_your_own_trainer)

</details>

______________________________________________________________________

<div align="center">
    <a href="https://lightning.ai/docs/fabric/stable/">Read the Lightning Fabric docs</a>
</div>

______________________________________________________________________

## Continuous Integration

Lightning is rigorously tested across multiple CPUs and GPUs and against major Python and PyTorch versions.

###### \*Codecov is > 90%+ but build delays may show less

<details>
  <summary>Current build statuses</summary>

<center>

|       System / PyTorch ver.        |                                                   1.12                                                    |                                                   1.13                                                    |                                                    2.0                                                    |                                                       2.1                                                        |
| :--------------------------------: | :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
|         Linux py3.9 [GPUs]         |                                                                                                           |                                                                                                           |                                                                                                           | ![Build Status](https://dev.azure.com/Lightning-AI/lightning/_apis/build/status%2Flightning-fabric%20%28GPUs%29) |
|  Linux (multiple Python versions)  | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) |    ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg)     |
|   OSX (multiple Python versions)   | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) |    ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg)     |
| Windows (multiple Python versions) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) | ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg) |    ![Test Fabric](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-fabric.yml/badge.svg)     |

</center>
</details>

______________________________________________________________________

# Getting started

## Install Lightning

<details>

<summary>Prerequisites</summary>

> TIP: We strongly recommend creating a virtual environment first.
> Don’t know what this is? Follow our [beginner guide here](https://lightning.ai/docs/stable/install/installation.html).

- Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x, ...)

</details>

```bash
pip install -U lightning
```

## Convert your PyTorch to Fabric

1. Create the `Fabric` object at the beginning of your training code.

   ```
   import Lightning as L

   fabric = L.Fabric()
   ```

1. Call `setup()` on each model and optimizer pair and `setup_dataloaders()` on all your data loaders.

   ```
   model, optimizer = fabric.setup(model, optimizer)
   dataloader = fabric.setup_dataloaders(dataloader)
   ```

1. Remove all `.to` and `.cuda` calls -> Fabric will take care of it.

   ```diff
   - model.to(device)
   - batch.to(device)
   ```

1. Replace `loss.backward()` by `fabric.backward(loss)`.

   ```diff
   - loss.backward()
   + fabric.backward(loss)
   ```

1. Run the script from the terminal with

   ```bash
   lightning run model path/to/train.py
   ```

or use the launch() method in a notebook. Learn more about [launching distributed training](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html).

______________________________________________________________________

# FAQ

## When to use Fabric?

- **Minimum code changes**- You want to scale your PyTorch model to use multi-GPU or use advanced strategies like DeepSpeed without having to refactor. You don’t care about structuring your code- you just want to scale it as fast as possible.
- **Maximum control**- Write your own training and/or inference logic down to the individual optimizer calls. You aren’t forced to conform to a standardized epoch-based training loop like the one in Lightning Trainer. You can do flexible iteration based training, meta-learning, cross-validation and other types of optimization algorithms without digging into framework internals. This also makes it super easy to adopt Fabric in existing PyTorch projects to speed-up and scale your models without the compromise on large refactors. Just remember: With great power comes a great responsibility.
- **Maximum flexibility**- You want to have full control over your entire training- in Fabric all features are opt-in, and it provides you with a tool box of primitives so you can build your own Trainer.

## When to use the [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)?

- You want to have all the engineering boilerplate handled for you - dozens of features like checkpointing, logging and early stopping out of the box. Less hassle, less error prone, easy to try different techniques and features.
- You want to have good defaults chosen for you - so you can have a better starting point.
- You want your code to be modular, readable and well structured - easy to share between projects and with collaborators.

## Can I use Fabric with my LightningModule or Lightning Callback?

Yes :) Fabric works with PyTorch LightningModules and Callbacks, so you can choose how to structure your code and reuse existing models and callbacks as you wish. Read more [here](https://lightning.ai/docs/fabric/stable/fundamentals/code_structure.html).

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/continuum.png" width="800px">

______________________________________________________________________

# Examples

- [GAN](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/dcgan)
- [Meta learning](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/meta_learning)
- [Reinforcement learning](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/reinforcement_learning)
- [K-Fold cross validation](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/kfold_cv)

______________________________________________________________________

## Asking for help

If you have any questions please:

1. [Read the docs](https://lightning.ai/docs/fabric).
1. [Ask a question in our forum](https://lightning.ai/forums/).
1. [Join our discord community](https://discord.com/invite/tfXFetEZxv).
