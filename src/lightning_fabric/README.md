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
  <a href="https://www.pytorchlightning.ai/community">Slack</a>
</p>

</div>

## Maximum flexibility, minimum code changes

With just a few code changes, run any PyTorch model on any distributed hardware, no boilerplate!

- Easily switch from running on CPU to GPU (Apple Silicon, CUDA, …), TPU, multi-GPU or even multi-node training
- Use state-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed) and mixed precision out of the box
- All the device logic boilerplate is handled for you
- Designed with multi-billion parameter models in mind
- Build your own custom Trainer using Fabric primitives for training checkpointing, logging, and more

```diff
+ import lightning as L

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, Dataset

  class PyTorchModel(nn.Module):
      ...

  class PyTorchDataset(Dataset):
      ...

+ fabric = L.Fabric(accelerator="cuda", devices=8, strategy="ddp")
+ fabric.launch()

- device = "cuda" if torch.cuda.is_available() else "cpu
  model = PyTorchModel(...)
  optimizer = torch.optim.SGD(model.parameters())
+ model, optimizer = fabric.setup(model, optimizer)
  dataloader = DataLoader(PyTorchDataset(...), ...)
+ dataloader = fabric.setup_dataloaders(dataloader)
  model.train()

  for epoch in range(num_epochs):
      for batch in dataloader:
          input, target = batch
-         input, target = input.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
-         loss.backward()
+         fabric.backward(loss)
          optimizer.step()
          lr_scheduler.step()
```
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

2. Call `setup()` on each model and optimizer pair and `setup_dataloaders()` on all your data loaders.

```
model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)
```

3. Remove all `.to` and `.cuda` calls -> Fabric will take care of it.

```diff
- model.to(device)
- batch.to(device)
```

4.  Replace `loss.backward()` by `fabric.backward(loss)`.

```diff
- loss.backward()
+ fabric.backward(loss)
```

4.  Run the script from the terminal with

```bash
lightning run model path/to/train.py
```
or use the launch() method in a notebook. Learn more about [launching distributed training](https://pytorch-lightning.readthedocs.io/en/stable/fabric/fundamentals/launch.html).

______________________________________________________________________


# FAQ

## When to use Fabric?

- **Minimum code changes**- You want to scale your PyTorch model to use multi-GPU or use advanced strategies like DeepSpeed without having to refactor. You don’t care about structuring your code- you just want to scale it as fast as possible.
- **Maxmium control**- Write your own training and/or inference logic down to the individual optimizer calls. You aren’t forced to conform to a standardized epoch-based training loop like the one in Lightning Trainer. You can do flexible iteration based training, meta-learning, cross-validation and other types of optimization algorithms without digging into framework internals. This also makes it super easy to adopt Fabric in existing PyTorch projects to speed-up and scale your models without the compromise on large refactors. Just remember: With great power comes a great responsibility.
- **Maxmium flexibility**- You want to have full control over your entire training- in Fabric all features are opt-in, and it provides you with a tool box of primitives so you can build your own Trainer.

## When to use the [Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)?

- You want to have all the engineering boilerplate handled for you - dozens of features like checkpointing, logging and early stopping out of the box. Less hassle, less error prone, easy to try different techniques and features.
- You want to have good defaults chosen for you - so you can have a better starting point.
- You want your code to be modular, readable and well structured - easy to share between projects and with collaborators.

## Can I use Fabric with my LightningModule or Lightning Callback?

Yes :) Fabric works with PyTorch LightningModules and Callbacks, so you can choose how to structure your code and reuse existing models and callbacks as you wish. Read more [here](https://pytorch-lightning.readthedocs.io/en/stable/fabric/fundamentals/code_structure.html).

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/continuum.png" width="800px">

______________________________________________________________________


# Examples

- GAN
- Meta learning
- Large language models
- Reinforcement learning
- Active learning

______________________________________________________________________


## Asking for help

If you have any questions please:

1. [Read the docs](https://pytorch-lightning.readthedocs.io/en/stable/fabric).
1. [Ask a question in our forum](https://lightning.ai/forums/).
1. [Join our discord community](https://discord.com/invite/tfXFetEZxv).
