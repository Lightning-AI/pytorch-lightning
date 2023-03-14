<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pl-public-data.s3.amazonaws.com/assets_lightning/LightningDark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://pl-public-data.s3.amazonaws.com/assets_lightning/LightningLight.png">
    <img alt="Lightning" src="hhttps://pl-public-data.s3.amazonaws.com/assets_lightning/LightningDark.png" width="400" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>

**The Deep Learning framework to train, deploy, and ship AI products Lightning fast.**

**NEW- Ligthning 2.0 is featuring a clean and stable API!**
______________________________________________________________________

<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="src/pytorch_lightning/README.md">PyTorch Lightning</a> •
  <a href="src/lightning_fabric/README.md">Fabric</a> •
  <a href="src/lightning_app/README.md">Lightning Apps</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/generated/CONTRIBUTING.html">Contribute</a> •
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)
[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)
[![codecov](https://codecov.io/gh/Lightning-AI/lightning/branch/master/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/lightning)

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://www.pytorchlightning.ai/community)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/Lightning-AI/lightning/badge)](https://www.codefactor.io/repository/github/Lightning-AI/lightning)
-->

</div>
______________________________________________________________________

## Train and deploy with PyTorch Lightning

PyTorch Lightning is just organized PyTorch- Lightning disentangles PyTorch code to decouple the science from the engineering.
![PT to PL](docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif)

<details>
  <summary>How to use PyTorch Lightning</summary>

  ### Step 1: Add these imports

  ```python
  import lightning as L
  
  import os
  import torch
  from torch import nn
  import torch.nn.functional as F
  from torchvision.datasets import MNIST
  from torch.utils.data import DataLoader, random_split
  from torchvision import transforms
  ```

  ### Step 2: Define a LightningModule (nn.Module subclass)

  A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

  ```python
  class LitAutoEncoder(L.LightningModule):
      def __init__(self):
          super().__init__()
          self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
          self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

      def forward(self, x):
          # in lightning, forward defines the prediction/inference actions
          embedding = self.encoder(x)
          return embedding

      def training_step(self, batch, batch_idx):
          # training_step defines the train loop. It is independent of forward
          x, y = batch
          x = x.view(x.size(0), -1)
          z = self.encoder(x)
          x_hat = self.decoder(z)
          loss = F.mse_loss(x_hat, x)
          self.log("train_loss", loss)
          return loss

      def configure_optimizers(self):
          optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
          return optimizer
  ```

  **Note: Training_step defines the training loop. Forward defines how the LightningModule behaves during inference/prediction.**

  ### Step 3: Train!

  ```python
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
  train, val = random_split(dataset, [55000, 5000])

  autoencoder = LitAutoEncoder()
  trainer = L.Trainer()
  trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
  ```

  ## Advanced features

  Lightning has over [40+ advanced features](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags) designed for professional AI research at scale.

  Here are some examples:

  <div align="center">
    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/features_2.jpg" max-height="600px">
  </div>

  <details>
    <summary>Highlighted feature code snippets</summary>

  ```python
  # 8 GPUs
  # no code changes needed
  trainer = Trainer(max_epochs=1, accelerator="gpu", devices=8)

  # 256 GPUs
  trainer = Trainer(max_epochs=1, accelerator="gpu", devices=8, num_nodes=32)
  ```

  <summary>Train on TPUs without code changes</summary>

  ```python
  # no code changes needed
  trainer = Trainer(accelerator="tpu", devices=8)
  ```

  <summary>16-bit precision</summary>

  ```python
  # no code changes needed
  trainer = Trainer(precision=16)
  ```

  <summary>Experiment managers</summary>

  ```python
  from lightning import loggers

  # tensorboard
  trainer = Trainer(logger=TensorBoardLogger("logs/"))

  # weights and biases
  trainer = Trainer(logger=loggers.WandbLogger())

  # comet
  trainer = Trainer(logger=loggers.CometLogger())

  # mlflow
  trainer = Trainer(logger=loggers.MLFlowLogger())

  # neptune
  trainer = Trainer(logger=loggers.NeptuneLogger())

  # ... and dozens more
  ```

  <summary>EarlyStopping</summary>

  ```python
  es = EarlyStopping(monitor="val_loss")
  trainer = Trainer(callbacks=[es])
  ```

  <summary>Checkpointing</summary>

  ```python
  checkpointing = ModelCheckpoint(monitor="val_loss")
  trainer = Trainer(callbacks=[checkpointing])
  ```

  <summary>Export to torchscript (JIT) (production use)</summary>

  ```python
  # torchscript
  autoencoder = LitAutoEncoder()
  torch.jit.save(autoencoder.to_torchscript(), "model.pt")
  ```

  <summary>Export to ONNX (production use)</summary>

  ```python
  # onnx
  with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
      autoencoder = LitAutoEncoder()
      input_sample = torch.randn((1, 64))
      autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
      os.path.isfile(tmpfile.name)
  ```

  </details>

  ### Pro-level control of optimization (advanced users)

  For complex/professional level work, you have optional full control of the optimizers.

  ```python
  class LitAutoEncoder(L.LightningModule):
      def __init__(self):
          super().__init__()
          self.automatic_optimization = False

      def training_step(self, batch, batch_idx):
          # access your optimizers with use_pl_optimizer=False. Default is True
          opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

          loss_a = ...
          self.manual_backward(loss_a, opt_a)
          opt_a.step()
          opt_a.zero_grad()

          loss_b = ...
          self.manual_backward(loss_b, opt_b, retain_graph=True)
          self.manual_backward(loss_b, opt_b)
          opt_b.step()
          opt_b.zero_grad()
  ```

  ______________________________________________________________________

  ## Advantages over unstructured PyTorch

  - Models become hardware agnostic
  - Code is clear to read because engineering code is abstracted away
  - Easier to reproduce
  - Make fewer mistakes because lightning handles the tricky engineering
  - Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
  - Lightning has dozens of integrations with popular machine learning tools.
  - [Tested rigorously with every new PR](https://github.com/Lightning-AI/lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
  - Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

  ______________________________________________________________________

  ## Examples

  ###### Self-supervised Learning

  - [CPC transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#cpc-transforms)
  - [Moco v2 tranforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#moco-v2-transforms)
  - [SimCLR transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#simclr-transforms)

  ###### Convolutional Architectures

  - [GPT-2](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#gpt-2)
  - [UNet](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#unet)

  ###### Reinforcement Learning

  - [DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#dqn-loss)
  - [Double DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#double-dqn-loss)
  - [Per DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#per-dqn-loss)

  ###### GANs

  - [Basic GAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#basic-gan)
  - [DCGAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#dcgan)

  ###### Classic ML

  - [Logistic Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#logistic-regression)
  - [Linear Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#linear-regression)

  ______________________________________________________________________

  ### [Read more about PyTorch Lightning](src/pytorch_lightning/README.md)

</details>

______________________________________________________________________

## Scale PyTorch With Lightning Fabric

Fabric allows you to scale any PyTorch model to distributed machines while maintianing full control over your training loop. Just add a few lines of code and run on any device!
Use this library for complex tasks like reinforcement learning, active learning, and transformers without losing control over your training code.

<div align="center">
    <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/continuum.png" width="80%">
</div>


<details>
  <summary>Learn more about Fabric</summary>
  
  
  
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


### [Read more about Fabric](src/fabric/README.md)
  
</details>


--------------------
## Build AI products with Lightning Apps

Once you're done building models, publish a paper demo or build a full production end-to-end ML system with Lightning Apps. Lightning Apps remove the cloud infrastructure boilerplate so you can focus on solving the research or business problems. Lightning Apps can run on the Lightning Cloud, your own cluster or a private cloud.

[Browse available Lightning apps here](https://lightning.ai/)

<div align="center">
    <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning-apps-teaser.png" width="80%">
</div>

<details>
  <summary>Learn more about apps</summary>
  
 Build machine learning components that can plug into existing ML workflows. A Lightning component organizes arbitrary code to run on the cloud, manage its own infrastructure, cloud costs, networking, and more. Focus on component logic and not engineering.

  Use components on their own, or compose them into full-stack AI apps with our next-generation Lightning orchestrator. to package your code into Lightning components which can plug into your existing ML workflows.

  
  ## Run your first Lightning App

  1. Install a simple training and deployment app by typing:

  ```bash
  # install lightning
  pip install lightning
  
  lightning install app lightning/quick-start
  ```

  2. If everything was successful, move into the new directory:

  ```bash
  cd lightning-quick-start
  ```

  3. Run the app locally

  ```bash
  lightning run app app.py
  ```

  4. Alternatively, run it on the public Lightning Cloud to share your app!

  ```bash
  lightning run app app.py --cloud
  ```
  
  Apps run the same on the cloud and locally on your choice of hardware.

  ## run the app on the --cloud
  lightning run app app.py --setup --cloud

  ### [Learn more about Lightning Apps](src/lightning_app/README.md)
  
</details>

______________________________________________________________________

## Continuous Integration

Lightning is rigorously tested across multiple CPUs, GPUs, TPUs, IPUs, and HPUs and against major Python and PyTorch versions.

###### \*Codecov is > 90%+ but build delays may show less

<details>
  <summary>Current build statuses</summary>

<center>

|       System / PyTorch ver.        |                                                                                              1.11                                                                                               |                                                                                                              1.12                                                                                                               | 1.13                                                                                                                                                                                                                            | 2.0  |
| :--------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
|        Linux py3.9 \[GPUs\]        |                                                                                                -                                                                                                | [![Build Status](<https://dev.azure.com/Lightning-AI/lightning/_apis/build/status/pytorch-lightning%20(GPUs)?branchName=master>)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=24&branchName=master) | [![Build Status](<https://dev.azure.com/Lightning-AI/lightning/_apis/build/status/pytorch-lightning%20(GPUs)?branchName=master>)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=24&branchName=master) | Soon |
|        Linux py3.9 \[TPUs\]        |                                                                                                -                                                                                                |                     [![Test PyTorch - TPU](https://github.com/Lightning-AI/lightning/actions/workflows/tpu-tests.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/tpu-tests.yml)                     |                                                                                                                                                                                                                                 | Soon |
|        Linux py3.8 \[IPUs\]        |                                                                                                -                                                                                                |                                                                                                                -                                                                                                                | [![Build Status](<https://dev.azure.com/Lightning-AI/lightning/_apis/build/status/pytorch-lightning%20(IPUs)?branchName=master>)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=25&branchName=master) | Soon |
|        Linux py3.8 \[HPUs\]        |                                                                                                -                                                                                                |                                                                                                                -                                                                                                                | [![Build Status](<https://dev.azure.com/Lightning-AI/lightning/_apis/build/status/pytorch-lightning%20(HPUs)?branchName=master>)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=26&branchName=master) | Soon |
|  Linux (multiple Python versions)  | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | Soon |
|   OSX (multiple Python versions)   | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | Soon |
| Windows (multiple Python versions) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml) |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | Soon |

</center>
</details>

______________________________________________________________________

## Install

Simple installation from PyPI

```bash
pip install lightning
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Other installation options</summary>
    <!-- following section will be skipped from PyPI description -->

#### Install with optional dependencies

```bash
pip install lightning['extra']
```

#### Conda

```bash
conda install lightning -c conda-forge
```

#### Install stable version

Install future release from the source

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
```

#### Install bleeding-edge

Install nightly from the source (no guarantees)

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
```

or from testing PyPI

```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

</details>
<!-- end skipping PyPI description -->


______________________________________________________________________

## Community

The lightning community is maintained by

- [10+ core contributors](https://pytorch-lightning.readthedocs.io/en/latest/governance.html) who are all a mix of professional engineers, Research Scientists, and Ph.D. students from top AI labs.
- 590+ active community contributors.

Want to help us build Lightning and reduce boilerplate for thousands of researchers? [Learn how to make your first contribution here](https://pytorch-lightning.readthedocs.io/en/stable/generated/CONTRIBUTING.html)

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help

If you have any questions please:

1. [Read the docs](https://pytorch-lightning.rtfd.io/en/latest).
1. [Search through existing Discussions](https://github.com/Lightning-AI/lightning/discussions), or [add a new question](https://github.com/Lightning-AI/lightning/discussions/new)
1. [Join our discord](https://discord.com/invite/tfXFetEZxv).
