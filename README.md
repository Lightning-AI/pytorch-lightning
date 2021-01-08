<div align="center">

<img src="docs/source/_images/logos/lightning_logo-name.png" width="400px">


**The lightweight PyTorch wrapper for high-performance AI research.
Scale your models, not the boilerplate.**

---

<p align="center">
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/">Docs</a> •
  <a href="#examples">Examples</a> •
  <a href="#community">Community</a> •
  <a href="#grid-ai">Grid AI</a> •
  <a href="#licence">Licence</a>
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)
[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)
[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning)

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![Discourse status](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforums.pytorchlightning.ai)](https://forums.pytorchlightning.ai/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Next Release](https://img.shields.io/badge/Next%20Release-Nov%2021-<COLOR>.svg)](https://shields.io/)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning)
-->
</div>

###### *Codecov is > 90%+ but build delays may show less

---

## NEWS
[Dec 2020 - Read about how Facebook uses Lightning to standardize deep learning across research and production teams](https://ai.facebook.com/blog/reengineering-facebook-ais-deep-learning-platforms-for-interoperability)

---

## PyTorch Lightning is just organized PyTorch
Lightning disentangles PyTorch code to decouple the science from the engineering.
![PT to PL](docs/source/_images/general/pl_quick_start_full_compressed.gif)

---

## Lightning Philosophy
Lightning is designed with these principles in mind:

Principle 1: Enable maximal flexibility.
Principle 2: Abstract away unnecessary boilerplate, but make it accessible when needed.
Principle 3: Systems should be self-contained (ie: optimizers, computation code, etc).
Principle 4: Deep learning code should be organized into 4 distinct categories.

  - Research code (the LightningModule).
  - Engineering code (you delete, and is handled by the Trainer).
  - Non-essential research code (logging, etc... this goes in Callbacks).
  - Data (use PyTorch Dataloaders or organize them into a LightningDataModule).

Once you do this, you can train on multiple-GPUs, TPUs, CPUs and even in 16-bit precision without changing your code!

Get started with our [2 step guide](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)

---

## Inference
Lightning is also designed for the fast inference AI researchers and production teams need to scale up things like BERT and self-supervised learning.
Lightning can automatically export to ONNX or TorchScript for those cases.

---

## Continuous Integration
<center>

| System / PyTorch ver. | 1.3 (min. req.)* | 1.4 | 1.5 | 1.6 | 1.7 (latest) | 1.8 (nightly) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conda py3.7 [linux] | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) |
| Linux py3.7 [GPUs**] | - | - | - | [![GPUs Status](http://104.154.220.231/api/badges/PyTorchLightning/pytorch-lightning/status.svg)](http://104.154.220.231/PyTorchLightning/pytorch-lightning) | - | - |
| Linux py3.{6,7} [TPUs***] | - | - | - | [![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22TPU+tests%22+branch%3Amaster) | [![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22TPU+tests%22+branch%3Amaster) | - |
| Linux py3.{6,7} | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |
| OSX py3.{6,7,8} | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |
| Windows py3.{6,7,8} | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |

- _\* `torch>=1.4` is the minimal pytorch version for Python 3.8_
- _\** tests run on two NVIDIA K80_
- _\*** tests run on Google GKE TPUv2/3_
- _TPU w/ py3.6/py3.7 means we support Colab and Kaggle env._

</center>

---

## How To Use

### Step 0: Install

Simple installation from PyPI
```bash
pip install pytorch-lightning
```
_To get full package experience you can install also all optional dependencies with `pytorch-lightning['extra']` or for CPU users with `pytorch-lightning['cpu-extra']`._

From Conda
```bash
conda install pytorch-lightning -c conda-forge
```

<!-- following section will be skipped from PyPI description -->

#### Install bleeding-edge - future 1.2

the actual status of 1.2 [nightly] is following:

![CI base testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20base%20testing/badge.svg?branch=release%2F1.2-dev&event=push)
![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=release%2F1.2-dev&event=push)
![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=release%2F1.2-dev&event=push)
![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=release%2F1.2-dev&event=push)
![Docs check](https://github.com/PyTorchLightning/pytorch-lightning/workflows/Docs%20check/badge.svg?branch=release%2F1.2-dev&event=push)

Install future release from the source (no guarantees)
```bash
pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@release/1.2-dev --upgrade
```
or nightly from testing PyPI
```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

<!-- end skipping PyPI description -->

### Step 1: Add these imports

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
```

### Step 2: Define a LightningModule (nn.Module subclass)
A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

```python
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
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
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

#### And without changing a single line of code, you could run on GPUs/TPUs
```python
# 8 GPUs
trainer = Trainer(max_epochs=1, gpus=8)

# 256 GPUs
trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)

# TPUs
trainer = Trainer(tpu_cores=8)
```

#### And even export for production via onnx or torchscript
```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")

# onnx
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
    autoencoder = LitAutoEncoder()
    input_sample = torch.randn((1, 64))
    autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

#### For advanced users, you can still own complex training loops

```python
class LitAutoEncoder(pl.LightningModule):
    def training_step(self, batch, batch_idx, opt_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        (opt_a, opt_b) = self.optimizers(use_pl_optimizer=True)

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
---

## Key Features

* Scale your models to run on any hardware (CPU, GPUs, TPUs) without changing your model
* Making code more readable by decoupling the research code from the engineering
* Easier to reproduce
* Less error prone by automating most of the training loop and tricky engineering
* Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
* Lightning has out-of-the-box integration with the popular logging/visualizing frameworks ([Tensorboard](https://pytorch.org/docs/stable/tensorboard.html), [MLFlow](https://mlflow.org/), [Neptune.ai](https://neptune.ai/), [Comet.ml](https://www.comet.ml/site/), [Wandb](https://www.wandb.com/)).
* [Tested rigorously with every new PR](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
* Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

### Lightning automates 40+ parts of DL/ML research
- GPU training
- Distributed GPU (cluster) training
- TPU training
- EarlyStopping
- Logging/Visualizing
- Checkpointing
- Experiment management
- [Full list here](https://pytorch-lightning.readthedocs.io/en/latest/#common-use-cases)

---

## Examples

###### Hello world
- [MNIST hello world](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb)
- [MNIST on TPUs](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/06-mnist-tpu-training.ipynb)

###### Contrastive Learning
- [BYOL](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#byol)
- [CPC v2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#cpc-v2)
- [Moco v2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#moco-v2)
- [SIMCLR](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr)

###### NLP
- [BERT](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb)
- [GPT-2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/convolutional.html#gpt-2)


###### Reinforcement Learning
- [DQN](https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dqn-models)
- [Dueling-DQN](https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dueling-dqn)
- [Reinforce](https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#reinforce)

###### Vision
- [GAN](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb)

###### Classic ML
- [Logistic Regression](https://pytorch-lightning-bolts.readthedocs.io/en/latest/classic_ml.html#logistic-regression)
- [Linear Regression](https://pytorch-lightning-bolts.readthedocs.io/en/latest/classic_ml.html#linear-regression)

---

## Community

The lightning community is maintained by
- [16 core contributors](https://pytorch-lightning.readthedocs.io/en/latest/governance.html) who are all a mix of professional engineers, Research Scientists, Ph.D. students from top AI labs.
- 280+ community contributors.

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help
If you have any questions please:
1. [Read the docs](https://pytorch-lightning.rtfd.io/en/latest/).
2. [Look it up in our forum (or add a new question)](https://forums.pytorchlightning.ai/)
2. [Search through the issues](https://github.com/PytorchLightning/pytorch-lightning/issues?utf8=%E2%9C%93&q=my++question).
3. [Join our slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A).
4. [Ask on stackoverflow](https://stackoverflow.com/questions/ask?guided=false) with the tag pytorch-lightning.

### Funding
Building open-source software with only a few part-time people is hard!

[We're venture funded](https://techcrunch.com/2020/10/08/grid-ai-raises-18-6m-series-a-to-help-ai-researchers-and-engineers-bring-their-models-to-production/)
and backed by some of the top VC funds in the world, [Index Ventures](https://www.indexventures.com/companies/), [Bain Capital Ventures](https://www.baincapitalventures.com/portfolio/), [First Minute Capital](https://firstminute.capital/companies).

Their funding ensures we can continue to build awesome tooling like Grid, give you around the clock support,
hire a full-time staff, attend conferences, and move faster through implementing features you request.

To supercharge your research and production work, visit our [Grid.ai platform](https://www.grid.ai/)

---

## Grid AI
Grid AI is our native platform for training models at scale on the cloud!

**Sign up for [early access here](https://www.grid.ai/)**

To use grid, take your regular command:

```
    python my_model.py --learning_rate 1e-6 --layers 2 --gpus 4
```

And change it to use the grid train command:

```
    grid train --grid_gpus 4 my_model.py --learning_rate 'uniform(1e-6, 1e-1, 20)' --layers '[2, 4, 8, 16]'
```

The above command will launch (20 * 4) experiments each running on 4 GPUs (320 GPUs!) - by making ZERO changes to
your code.

---

## Licence

Please observe the Apache 2.0 license that is listed in this repository. In addition
the Lightning framework is Patent Pending.

## BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊):

```bibtex
@article{falcon2019pytorch,
  title={PyTorch Lightning},
  author={Falcon, WA},
  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning},
  volume={3},
  year={2019}
}
```
