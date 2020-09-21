<div align="center">

![Logo](docs/source/_images/logos/lightning_logo.svg)

# PyTorch Lightning

**The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.**

<p align="center">
  <a href="https://www.youtube.com/watch?v=DbESHcCoWbM&t=2s">Masterclass</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/">Docs</a> •
  <a href="#examples">Examples</a> •
  <a href="#community">Community</a> •
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
[![Next Release](https://img.shields.io/badge/Next%20Release-Sep%2014-<COLOR>.svg)](https://shields.io/)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning)
-->
</div>

###### *Codecov is > 90%+ but build delays may show less

## PyTorch Lightning is just organized PyTorch
Lightning disentangles PyTorch code to decouple the science from the engineering.
![PT to PL](/docs/source/_images/general/pl_quick_start_full_compressed.gif)

---

## Lightning Philosophy
Lightning is designed with these principles in mind:

Principle 1: Enable maximal flexibility.
Principle 2: Abstract away unecessary boilerplate, but make it accessible when needed.
Principle 3: Systems should be self-contained (ie: optimizers, computation code, etc).

Principle 4: Finally, Deep learning code should be organized into 4 distinct categories

  - Research code (the LightningModule).
  - Engineering code (you delete, and is handled by the Trainer).
  - Non-essential research code (logging, etc... this goes in Callbacks).
  - Data (use PyTorch Dataloaders or organize them into a LightningDataModule).

Once you do this, you can train on multiple-GPUs, TPUs, CPUs and even in 16-bit precision without changing your code!

Get started with our [3 steps guide](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)

---
## Trending contributors

[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/0)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/0)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/1)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/1)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/2)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/2)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/3)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/3)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/4)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/4)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/5)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/5)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/6)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/6)
[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/images/7)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning/links/7)

---

## Continuous Integration
<center>

| System / PyTorch ver. | 1.3 (min. req.)* | 1.4 | 1.5 | 1.6 (latest) |
| :---: | :---: | :---: | :---: | :---: |
| Conda py3.7 [linux] | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) |
| Linux py3.7 [GPUs**] | - | - | - | [![Build Status](http://35.192.60.23/api/badges/PyTorchLightning/pytorch-lightning/status.svg)](http://35.192.60.23/PyTorchLightning/pytorch-lightning) |
| Linux py3.7 [TPUs***] | - | - | - | [![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22TPU+tests%22+branch%3Amaster) |
| Linux py3.6 / py3.7 / py3.8 | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) |
| OSX py3.6 / py3.7 | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) |
| Windows py3.6 / py3.7 / py3.8 | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) |

- _\* `torch>=1.4` is the minimal pytorch version for Python 3.8_
- _\** tests run on two NVIDIA K80_
- _\*** tests run on Google GKE TPUv2/3_

</center>

---

## How To Use

#### Setup step: Install
Simple installation from PyPI
```bash
pip install pytorch-lightning
```

From Conda
```bash
conda install pytorch-lightning -c conda-forge
```

Install bleeding-edge (no guarantees)   
```bash
pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
```

#### Setup step: Add these imports

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

#### Step 1: Define a LightningModule
A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

```python
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

#### Step 2: Train!

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

#### And without changing a single line of code, you could run on GPUs
```python
# 8 GPUs
trainer = Trainer(max_epochs=1, gpus=8)

# 256 GPUs
trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)
```

Or TPUs
```python
# Distributes TPU core training
trainer = Trainer(tpu_cores=8)

# Single TPU core training
trainer = Trainer(tpu_cores=[1])
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
[MNIST hello world](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL)  
[MNIST on TPUs](https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3)

###### Contrastive Learning
[BYOL](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#byol)    
[CPC v2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#cpc-v2)    
[Moco v2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#moco-v2)    
[SIMCLR](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr) 

###### NLP
[BERT](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=7uQVI-xv9Ddj)   
[GPT-2](https://pytorch-lightning-bolts.readthedocs.io/en/latest/convolutional.html#gpt-2) 


###### Reinforcement Learning
[DQN](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=NWvMLBDySQI5)   
[Dueling-DQN](https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dueling-dqn)   
[Reinforce](https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#reinforce)

###### Vision
[GAN](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=P0bSmCw57aV5)   

###### Classic ML
[Logistic Regression](https://pytorch-lightning-bolts.readthedocs.io/en/latest/classic_ml.html#logistic-regression)   
[Linear Regression](https://pytorch-lightning-bolts.readthedocs.io/en/latest/classic_ml.html#linear-regression)    

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
Building open-source software with only a few part-time people is hard! We've secured funding to make sure we can
hire a full-time staff, attend conferences, and move faster through implementing features you request.

Our goal is to build an incredible research platform and a big supportive community. Many open-source projects
have gone on to fund operations through things like support and special help for big corporations!

If you are one of these corporations, please feel free to reach out to will@pytorchlightning.ai!

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
