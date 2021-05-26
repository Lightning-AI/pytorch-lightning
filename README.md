<div align="center">

<img src="docs/source/_static/images/logo.png" width="400px">


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
  <a href="#license">License</a>
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)
[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)
[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning)

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning)
-->
</div>

###### *Codecov is > 90%+ but build delays may show less

---

## PyTorch Lightning is just organized PyTorch
Lightning disentangles PyTorch code to decouple the science from the engineering.
![PT to PL](docs/source/_static/images/general/pl_quick_start_full_compressed.gif)

---

## Lightning Design Philosophy
Lightning structures PyTorch code with these principles:

<div align="center">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/philosophies.jpg" max-height="250px">
</div>

Lightning forces the following structure to your code which makes it reusable and shareable:

- Research code (the LightningModule).
- Engineering code (you delete, and is handled by the Trainer).
- Non-essential research code (logging, etc... this goes in Callbacks).
- Data (use PyTorch DataLoaders or organize them into a LightningDataModule).

Once you do this, you can train on multiple-GPUs, TPUs, CPUs and even in 16-bit precision without changing your code!

Get started with our [2 step guide](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

---

## Continuous Integration
Lightning is rigorously tested across multiple GPUs, TPUs CPUs and against major Python and PyTorch versions.

<details>
  <summary>Current build statuses</summary>

  <center>

  | System / PyTorch ver. | 1.4 (min. req.) | 1.5 | 1.6 | 1.7 | 1.8 (latest) | 1.9 (nightly) |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | Conda py3.7 [linux] | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) | [![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22PyTorch+%26+Conda%22+branch%3Amaster) |
  | Linux py3.7 [GPUs**] | - | - | [![Build Status](https://dev.azure.com/PytorchLightning/pytorch-lightning/_apis/build/status/PL.pytorch-lightning%20(GPUs)?branchName=master)](https://dev.azure.com/PytorchLightning/pytorch-lightning/_build/latest?definitionId=6&branchName=master) | - | - | - |
  | Linux py3.{6,7} [TPUs***] | - | - | [![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22TPU+tests%22+branch%3Amaster) | - | [![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22TPU+tests%22+branch%3Amaster) | - |
  | Linux py3.{6,7,8,9} | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |
  | OSX py3.{6,7,8,9} | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |
  | Windows py3.{6,7,8,9} | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | - | [![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - |

  - _\** tests run on two NVIDIA P100_
  - _\*** tests run on Google GKE TPUv2/3_
  - _TPU py3.7 means we support Colab and Kaggle env._

  </center>
</details>

---

## How To Use

### Step 0: Install

Simple installation from PyPI
```bash
pip install pytorch-lightning
```

<!-- following section will be skipped from PyPI description -->
<details>
  <summary>Other installation options</summary>
    <!-- following section will be skipped from PyPI description -->

  #### Install with optional dependencies

  ```bash
  pip install pytorch-lightning['extra']
  ```

  #### Conda

  ```bash
  conda install pytorch-lightning -c conda-forge
  ```

  #### Install stable 1.3.x

  the actual status of 1.3 [stable] is following:

  ![CI base testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20base%20testing/badge.svg?branch=release%2F1.3.x&event=push)
  ![CI complete testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20complete%20testing/badge.svg?branch=release%2F1.3.x&event=push)
  ![PyTorch & Conda](https://github.com/PyTorchLightning/pytorch-lightning/workflows/PyTorch%20&%20Conda/badge.svg?branch=release%2F1.3.x&event=push)
  ![TPU tests](https://github.com/PyTorchLightning/pytorch-lightning/workflows/TPU%20tests/badge.svg?branch=release%2F1.3.x&event=push)
  ![Docs check](https://github.com/PyTorchLightning/pytorch-lightning/workflows/Docs%20check/badge.svg?branch=release%2F1.3.x&event=push)

  Install future release from the source
  ```bash
  pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@release/1.3.x --upgrade
  ```

  #### Install bleeding-edge - future 1.4

  Install nightly from the source (no guarantees)
  ```bash
  pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
  ```

  or from testing PyPI
  ```bash
  pip install -iU https://test.pypi.org/simple/ pytorch-lightning
  ```
</details>
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
        # training_step defines the train loop. It is independent of forward
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
  trainer = Trainer(max_epochs=1, gpus=8)

  # 256 GPUs
  trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)
  ```

  <summary>Train on TPUs without code changes</summary>

  ```python
  # no code changes needed
  trainer = Trainer(tpu_cores=8)
   ```

  <summary>16-bit precision</summary>

  ```python
  # no code changes needed
  trainer = Trainer(precision=16)
   ```

  <summary>Experiment managers</summary>

  ```python
  from pytorch_lightning import loggers

  # tensorboard
  trainer = Trainer(logger=TensorBoardLogger('logs/'))

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
  es = EarlyStopping(monitor='val_loss')
  trainer = Trainer(callbacks=[es])
   ```

  <summary>Checkpointing</summary>

  ```python
  checkpointing = ModelCheckpoint(monitor='val_loss')
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
  with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
      autoencoder = LitAutoEncoder()
      input_sample = torch.randn((1, 64))
      autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
      os.path.isfile(tmpfile.name)
   ```
</details>

### Pro-level control of training loops (advanced users)
For complex/professional level work, you have optional full control of the training loop and optimizers.

```python
class LitAutoEncoder(pl.LightningModule):
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
---

## Advantages over unstructured PyTorch

* Models become hardware agnostic
* Code is clear to read because engineering code is abstracted away
* Easier to reproduce
* Make fewer mistakes because lightning handles the tricky engineering
* Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
* Lightning has dozens of integrations with popular machine learning tools.
* [Tested rigorously with every new PR](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
* Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

---

## Examples

###### Hello world
- [MNIST hello world](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb)
- [MNIST on TPUs](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/06-mnist-tpu-training.ipynb)

###### Contrastive Learning
- [BYOL](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#byol)
- [CPC v2](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#cpc-v2)
- [Moco v2](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#moco-v2)
- [SIMCLR](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr)

###### NLP
- [BERT](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb)
- [GPT-2](https://lightning-bolts.readthedocs.io/en/latest/convolutional.html#gpt-2)


###### Reinforcement Learning
- [DQN](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dqn-models)
- [Dueling-DQN](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#dueling-dqn)
- [Reinforce](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html#reinforce)

###### Vision
- [GAN](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb)

###### Classic ML
- [Logistic Regression](https://lightning-bolts.readthedocs.io/en/latest/classic_ml.html#logistic-regression)
- [Linear Regression](https://lightning-bolts.readthedocs.io/en/latest/classic_ml.html#linear-regression)

---

## Community

The lightning community is maintained by
- [10+ core contributors](https://pytorch-lightning.readthedocs.io/en/latest/governance.html) who are all a mix of professional engineers, Research Scientists, and Ph.D. students from top AI labs.
- 400+ community contributors.

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help
If you have any questions please:
1. [Read the docs](https://pytorch-lightning.rtfd.io/en/latest).
2. [Search through existing Discussions](https://github.com/PyTorchLightning/pytorch-lightning/discussions), or [add a new question](https://github.com/PyTorchLightning/pytorch-lightning/discussions/new)
3. [Join our slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ).
### Funding
[We're venture funded](https://techcrunch.com/2020/10/08/grid-ai-raises-18-6m-series-a-to-help-ai-researchers-and-engineers-bring-their-models-to-production/) to make sure we can provide around the clock support, hire a full-time staff, attend conferences, and move faster through implementing features you request.

---

## Grid AI
Grid AI is our platform for training models at scale on the cloud!

**Sign up for our FREE community Tier [here](https://www.grid.ai/pricing/)**

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

Please observe the Apache 2.0 license that is listed in this repository.
In addition, the Lightning framework is Patent Pending.

## BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊) or [zenodo](https://zenodo.org/record/3828935#.YC45Lc9Khqs):

```bibtex
@article{falcon2019pytorch,
  title={PyTorch Lightning},
  author={Falcon, WA, et al.},
  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning},
  volume={3},
  year={2019}
}
```
