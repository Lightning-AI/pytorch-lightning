<div align="center">

![Logo](docs/source/_images/logos/lightning_logo.svg)

# PyTorch Lightning

**The lightweight PyTorch wrapper for ML researchers. Scale your models. Write less boilerplate.**

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#convert-your-pytorch">Convert your PyTorch</a> •
  <a href="#resources">Resources</a> •
  <a href="#community">Community</a> •
  <a href="#faq">FAQ</a> •
  <a href="#licence">Licence</a>
</p>


[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)
[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning)

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Next Release](https://img.shields.io/badge/Next%20Release-May%2029-<COLOR>.svg)](https://shields.io/)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning)
-->
</div>

## PyTorch Lightning is just organized PyTorch
![PT to PL](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_images/general/fast_2.gif)

Lightning is a way to organize your PyTorch code to decouple the science code from the engineering.
It's more of a PyTorch style-guide than a framework.

In Lightning, you organize your code into 3 distinct categories:

1. Research code (goes in the LightningModule).
2. Engineering code (delete the boilerplate, this part is automated by the Trainer).
3. Non-essential research code (logging, etc... this goes in Callbacks).

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
| Linux py3.6 / py3.7 / py3.8 | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) |
| OSX py3.6 / py3.7 | - | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) |
| Windows py3.6 / py3.7 / py3.8 | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) | - | - | [![CI testing](https://github.com/PyTorchLightning/pytorch-lightning/workflows/CI%20testing/badge.svg?event=push)](https://github.com/PyTorchLightning/pytorch-lightning/actions?query=workflow%3A%22CI+testing%22) 

- _\* `torch>=1.4` is the minimal pytorch version for Python 3.8_
- _\** tests run on two NVIDIA K80_
- _\*** tests run on Google GKE TPUv2/3_

</center>

---
### [PyTorch Lightning Masterclass (new lessons weekly)](https://www.youtube.com/watch?v=DbESHcCoWbM&list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2)
[![IMAGE ALT TEXT HERE](docs/source/_images/general/PTL101_youtube_thumbnail.jpg)](https://www.youtube.com/watch?v=DbESHcCoWbM&list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2)
---

## Key Features

* Scale your models to run on any hardware (CPU, GPUs, TPUs) without changing your model
* Making code more readable by decoupling the research code from the engineering
* Easier to reproduce
* Less error prone by automtaing most of the training loop and tricky engineering
* Prints easy to read warnings and errors, making it easier to get things right
* Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
* Lightning has out-of-the-box integration with the popular logging/visualizing frameworks ([Tensorboard](https://pytorch.org/docs/stable/tensorboard.html), [MLFlow](https://mlflow.org/), [Neptune.ai](https://neptune.ai/), [Comet.ml](https://www.comet.ml/site/), [Wandb](https://www.wandb.com/)).
* [Tested rigorously with every new PR](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/tests). We test every combination og PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
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

## How To Use

##### Install
Simple installation from PyPI
```bash
pip install pytorch-lightning
```

From Conda
```bash
conda install pytorch-lightning -c conda-forge
```

##### Here's a minimal example without a validation or test loop.

```python
# this is just a plain nn.Module with some structure

class LitClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# train!
train_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

model = LitClassifier()
trainer = Trainer(max_epochs=1)
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

#### Other examples:   
[MNIST hello world](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL)   
[GAN](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=P0bSmCw57aV5)   
[BERT](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=7uQVI-xv9Ddj)   
[DQN](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=NWvMLBDySQI5)   
[MNIST on TPUs](https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3)

---

## Convert your PyTorch

Here's how you would organize a realistic PyTorch project into Lightning.

![PT to PL](docs/source/_images/mnist_imgs/pt_to_pl.jpg)

In summary, you:

1. Define a [LightningModule](https://pytorch-lightning.rtfd.io/en/latest/lightning-module.html)
```python
    class LitSystem(pl.LightningModule):

        def __init__(self):
            super().__init__()
            # not the best model...
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            ...
```

2. Fit it with a [Trainer](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.html)
 ```python
 from pytorch_lightning import Trainer

 model = LitSystem()

 # most basic trainer, uses good defaults
 trainer = Trainer()
 trainer.fit(model)
 ```

### What types of research works?
Anything! Remember, that this is just organized PyTorch code.
The Training step defines the core complexity found in the training loop.

##### Could be as complex as a seq2seq

```python
# define what happens for training here
def training_step(self, batch, batch_idx):
    x, y = batch

    # define your own forward and loss calculation
    hidden_states = self.encoder(x)

    # even as complex as a seq-2-seq + attn model
    # (this is just a toy, non-working example to illustrate)
    start_token = '<SOS>'
    last_hidden = torch.zeros(...)
    loss = 0
    for step in range(max_seq_len):
        attn_context = self.attention_nn(hidden_states, start_token)
        pred = self.decoder(start_token, attn_context, last_hidden)
        last_hidden = pred
        pred = self.predict_nn(pred)
        loss += self.loss(last_hidden, y[step])

    #toy example as well
    loss = loss / max_seq_len
    return {'loss': loss}
```

##### Or as basic as CNN image classification

```python
# define what happens for validation here
def validation_step(self, batch, batch_idx):
    x, y = batch

    # or as basic as a CNN classification
    out = self(x)
    loss = my_loss(out, y)
    return {'loss': loss}
```

[Check out the COLAB demo here](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg)

## [Refactoring your PyTorch code + benefits + full walk-through](https://www.youtube.com/watch?v=QHww1JH7IDU)
[![Watch the video](docs/source/_images/general/tutorial_cover.jpg)](https://www.youtube.com/watch?v=QHww1JH7IDU)

---

## Resources

### Docs
- [master](https://pytorch-lightning.readthedocs.io/en/latest)
- [stable](https://pytorch-lightning.readthedocs.io/en/stable)
- [0.8.5](https://pytorch-lightning.readthedocs.io/en/0.8.5/)
- [0.8.4](https://pytorch-lightning.readthedocs.io/en/0.8.4/)
- [0.8.3](https://pytorch-lightning.readthedocs.io/en/0.8.3/)
- [0.8.1](https://pytorch-lightning.readthedocs.io/en/0.8.1/)
- [0.7.6](https://pytorch-lightning.readthedocs.io/en/0.7.6/) 


### Examples
Check out this awesome list of research papers and implementations done with Lightning.

- [Contextual Emotion Detection (DoubleDistilBert)](https://github.com/PyTorchLightning/emotion_transformer)
- [Generative Adversarial Network](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=TyYOdg8g77P0)
- [Hyperparameter optimization with Optuna](https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py)
- [Hyperparameter optimization with Ray Tune](https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html)
- [Image Inpainting using Partial Convolutions](https://github.com/ryanwongsa/Image-Inpainting)
- [MNIST on TPU](https://colab.research.google.com/drive/1-_LKx4HwAxl5M6xPJmqAAu444LTDQoa3#scrollTo=BHBz1_AnamN_)
- [NER (transformers, TPU, huggingface)](https://colab.research.google.com/drive/1dBN-wwYUngLYVt985wGs_OKPlK_ANB9D)
- [NeuralTexture (CVPR)](https://github.com/PyTorchLightning/neuraltexture)
- [Recurrent Attentive Neural Process](https://github.com/PyTorchLightning/attentive-neural-processes)
- [Siamese Nets for One-shot Image Recognition](https://github.com/PyTorchLightning/Siamese-Neural-Networks)
- [Speech Transformers](https://github.com/PyTorchLightning/speech-transformer-pytorch_lightning)
- [Transformers transfer learning (Huggingface)](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=yr7eaxkF-djf)
- [Transformers text classification](https://github.com/ricardorei/lightning-text-classification)
- [VAE Library of over 18+ VAE flavors](https://github.com/AntixK/PyTorch-VAE)
- [Transformers Question Answering (SQuAD)](https://github.com/tshrjn/Finetune-QA/)
- [Pytorch-Lightning + Microsoft NNI with Docker](https://github.com/davinnovation/pytorch-boilerplate)

### Tutorials
Check out our [introduction guide](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html) to get started.
Or jump straight into [our tutorials](https://pytorch-lightning.readthedocs.io/en/latest/#tutorials).

---

## Community

The lightning cimmunity is maintained by
- [15 core contributors](https://pytorch-lightning.readthedocs.io/en/latest/governance.html) who are all a mix of professional engineers, Research Scientists, Ph.D. students from top AI labs.
- 200+ community contributors.

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help
If you have any questions please:
1. [read the docs](https://pytorch-lightning.rtfd.io/en/latest/).
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

## FAQ

**Starting a new project?**
[Use our seed-project aimed at reproducibility!](https://github.com/PytorchLightning/pytorch-lightning-conference-seed)

**Why lightning?**
Although your research/production project might start simple, once you add things like GPU AND TPU training, 16-bit precision, etc, you end up spending more time engineering than researching. Lightning automates AND rigorously tests those parts for you.

**Who is Lightning for?**
- Professional researchers
- Ph.D. students
- Corporate production teams

If you're just getting into deep learning, we recommend you learn PyTorch first! Once you've implemented a few models, come back and use all the advanced features of Lightning :)

**What does lightning control for me?**
Everything in Blue!
This is how lightning separates the science (red) from engineering (blue).

![Overview](docs/source/_images/general/pl_overview.gif)

**How much effort is it to convert?**
If your code is not a huge mess you should be able to organize it into a LightningModule in less than 1 hour.
If your code IS a mess, then you needed to clean up anyhow ;)

[Check out this step-by-step guide](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).
[Or watch this video](https://www.youtube.com/watch?v=QHww1JH7IDU).

**How flexible is it?**
As you see, you're just organizing your PyTorch code - there's no abstraction.

And for the stuff that the Trainer abstracts out, you can [override any part](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#extensibility) you want to do things like implement your own distributed training, 16-bit precision, or even a custom backward pass.

For example, here you could do your own backward pass without worrying about GPUs, TPUs or 16-bit since we already handle it.

```python
class LitModel(LightningModule):
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                     second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
      optimizer.step()

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
      optimizer.zero_grad()
```

For anything else you might need, we have an extensive [callback system](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#callbacks) you can use to add arbitrary functionality not implemented by our team in the Trainer.

**How does performance compare with vanilla PyTorch?**
We have tests to ensure we get the EXACT same results in under 600 ms difference per epoch. In reality, lightning adds about a 300 ms overhead per epoch.
[Check out the parity tests here](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/benchmarks).

Overall, Lightning guarantees rigorously tested, correct, modern best practices for the automated parts.

**Why was Lightning created?**
Lightning has 3 goals in mind:

1. Maximal flexibility while abstracting out the common boilerplate across research projects.
2. Reproducibility. If all projects use the LightningModule template, it will be much much easier to understand what's going on and where to look! It will also mean every implementation follows a standard format.
3. Democratizing PyTorch power-user features. Distributed training? 16-bit? know you need them but don't want to take the time to implement? All good... these come built into Lightning.

**How does Lightning compare with Ignite and fast.ai?**
[Here's a thorough comparison](https://medium.com/@_willfalcon/pytorch-lightning-vs-pytorch-ignite-vs-fast-ai-61dc7480ad8a).

**Is this another library I have to learn?**
Nope! We use pure Pytorch everywhere and don't add unnecessary abstractions!

**Are there plans to support Python 2?**
Nope.

**Are there plans to support virtualenv?**
Nope. Please use anaconda or miniconda.
```bash
conda activate my_env
pip install pytorch-lightning
```

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
  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning Cited by},
  volume={3},
  year={2019}
}
```
