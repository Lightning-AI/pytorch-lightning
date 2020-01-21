<div align="center">

![Logo](docs/source/_static/images/lightning_logo_small.png)

# PyTorch Lightning

**The lightweight PyTorch wrapper for ML researchers. Scale your models. Write less boilerplate.**


[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Build Status](https://travis-ci.org/PytorchLightning/pytorch-lightning.svg?branch=master)](https://travis-ci.org/PytorchLightning/pytorch-lightning)
[![Build status](https://ci.appveyor.com/api/projects/status/NEW-PROJECT-ID?svg=true)](https://ci.appveyor.com/project/PytorchLightning/pytorch-lightning)
[![Coverage](docs/source/_static/images/coverage.svg)](https://github.com/PytorchLightning/pytorch-lightning/tree/master/tests#running-coverage)
[![CodeFactor](https://www.codefactor.io/repository/github/borda/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/borda/pytorch-lightning)    

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=latest)](https://pytorch-lightning.readthedocs.io/en/latest)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/enQtODU5ODIyNTUzODQwLTFkMDg5Mzc1MDBmNjEzMDgxOTVmYTdhYjA1MDdmODUyOTg2OGQ1ZWZkYTQzODhhNzdhZDA3YmNhMDhlMDY4YzQ)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Next Release](https://img.shields.io/badge/Next%20Release-Feb%206-<COLOR>.svg)](https://shields.io/)

<!-- 
removed until codecov badge isn't empy. likely a config error showing nothing on master.
[![codecov](https://codecov.io/gh/Borda/pytorch-lightning/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/pytorch-lightning)
-->

</div>

Simple installation from PyPI
```bash
pip install pytorch-lightning  
```

## Docs   
[jan 20, 2020]   

**[Old docs (some links might be broken)](https://pytorch-lightning.readthedocs.io/en/stable)
###### As a temporary hack, when you get the 404, replace williamfalcon.github.io with pytorchlightning.github.io.   

**[New docs, CURRENTLY DEBUGING](https://pytorch-lightning.rtfd.io/en/latest)**   


## Demo  
[Copy and run this COLAB!](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg)

## What is it?  
Lightning is a very lightweight wrapper on PyTorch that decouples the science code from the engineering code. It's more of a style-guide than a framework. By refactoring your code, we can automate most of the non-research code.  

To use Lightning, simply refactor your research code into the [LightningModule](https://github.com/PytorchLightning/pytorch-lightning#how-do-i-do-use-it) format (the science) and Lightning will automate the rest (the engineering). Lightning guarantees tested, correct, modern best practices for the automated parts.

- If you are a researcher, Lightning is infinitely flexible, you can modify everything down to the way .backward is called or distributed is set up. 
- If you are a scientist or production team, lightning is very simple to use with best practice defaults.

## What does lightning control for me?   

Everything in Blue!   
This is how lightning separates the science (red) from the engineering (blue).

![Overview](docs/source/_static/images/pl.gif)

## How much effort is it to convert?
You're probably tired of switching frameworks at this point. But it is a very quick process to refactor into the Lightning format (ie: hours). [Check out this tutorial](https://towardsdatascience.com/how-to-refactor-your-pytorch-code-to-get-these-42-benefits-of-pytorch-lighting-6fdd0dc97538)

## Starting a new project?   
[Use our seed-project aimed at reproducibility!](https://github.com/PytorchLightning/pytorch-lightning-conference-seed)     

## Why do I want to use lightning?
Every research project starts the same, a model, a training loop, validation loop, etc. As your research advances, you're likely to need distributed training, 16-bit precision, checkpointing, gradient accumulation, etc.   

Lightning sets up all the boilerplate state-of-the-art training for you so you can focus on the research.   

---
 
## README Table of Contents        
- [How do I use it](https://github.com/PytorchLightning/pytorch-lightning#how-do-i-do-use-it)     
- [What lightning automates](https://github.com/PytorchLightning/pytorch-lightning#what-does-lightning-control-for-me)    
- [Tensorboard integration](https://github.com/PytorchLightning/pytorch-lightning#tensorboard)    
- [Lightning features](https://github.com/PytorchLightning/pytorch-lightning#lightning-automates-all-of-the-following-each-is-also-configurable)    
- [Examples](https://github.com/PytorchLightning/pytorch-lightning#examples)    
- [Tutorials](https://github.com/PytorchLightning/pytorch-lightning#tutorials)
- [Contributing](https://github.com/PytorchLightning/pytorch-lightning/blob/master/.github/CONTRIBUTING.md)
- [Bleeding edge install](https://github.com/PytorchLightning/pytorch-lightning#bleeding-edge)   
- [Lightning Design Principles](https://github.com/PytorchLightning/pytorch-lightning#lightning-design-principles)   
- [Asking for help](https://github.com/PytorchLightning/pytorch-lightning#asking-for-help)
- [FAQ](https://github.com/PytorchLightning/pytorch-lightning#faq)    

---

## How do I do use it?   
Think about Lightning as refactoring your research code instead of using a new framework. The research code goes into a [LightningModule](https://pytorch-lightning.rtfd.io/en/latest/LightningModule/RequiredTrainerInterface/) which you fit using a Trainer.

The LightningModule defines a *system* such as seq-2-seq, GAN, etc... It can ALSO define a simple classifier such as the example below.     

To use lightning do 2 things:  
1. [Define a LightningModule](https://pytorch-lightning.rtfd.io/en/latest/LightningModule/RequiredTrainerInterface/)
**WARNING:** This syntax is for version 0.5.0+ where abbreviations were removed.
    ```python
    import os
    
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms
    
    import pytorch_lightning as pl
    
    class CoolSystem(pl.LightningModule):
    
        def __init__(self):
            super(CoolSystem, self).__init__()
            # not the best model...
            self.l1 = torch.nn.Linear(28 * 28, 10)
    
        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))
    
        def training_step(self, batch, batch_idx):
            # REQUIRED
            x, y = batch
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}
    
        def validation_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'val_loss': F.cross_entropy(y_hat, y)}
    
        def validation_end(self, outputs):
            # OPTIONAL
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
            
        def test_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'test_loss': F.cross_entropy(y_hat, y)}
    
        def test_end(self, outputs):
            # OPTIONAL
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            tensorboard_logs = {'test_loss': avg_loss}
            return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
        def configure_optimizers(self):
            # REQUIRED
            # can return multiple optimizers and learning_rate schedulers
            # (LBFGS it is automatically supported, no need for closure function)
            return torch.optim.Adam(self.parameters(), lr=0.02)
    
        @pl.data_loader
        def train_dataloader(self):
            # REQUIRED
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
    
        @pl.data_loader
        def val_dataloader(self):
            # OPTIONAL
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
    
        @pl.data_loader
        def test_dataloader(self):
            # OPTIONAL
            return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
    ```
2. Fit with a [trainer](https://pytorch-lightning.rtfd.io/en/latest/Trainer/)    
    ```python
    from pytorch_lightning import Trainer
    
    model = CoolSystem()
    
    # most basic trainer, uses good defaults
    trainer = Trainer()    
    trainer.fit(model)   
    ```

Trainer sets up a tensorboard logger, early stopping and checkpointing by default (you can modify all of them or
use something other than tensorboard).   

Here are more advanced examples
```python   
# train on cpu using only 10% of the data (for demo purposes)
trainer = Trainer(max_epochs=1, train_percent_check=0.1)

# train on 4 gpus (lightning chooses GPUs for you)
# trainer = Trainer(max_epochs=1, gpus=4, distributed_backend='ddp')  

# train on 4 gpus (you choose GPUs)
# trainer = Trainer(max_epochs=1, gpus=[0, 1, 3, 7], distributed_backend='ddp')   

# train on 32 gpus across 4 nodes (make sure to submit appropriate SLURM job)
# trainer = Trainer(max_epochs=1, gpus=8, num_gpu_nodes=4, distributed_backend='ddp')

# train (1 epoch only here for demo)
trainer.fit(model)

# view tensorboard logs 
logging.info(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
logging.info('and going to http://localhost:6006 on your browser')
```

When you're all done you can even run the test set separately.   
```python
trainer.test()
```

**Could be as complex as seq-2-seq + attention**    

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

**Or as basic as CNN image classification**      

```python
# define what happens for validation here
def validation_step(self, batch, batch_idx):    
    x, y = batch
    
    # or as basic as a CNN classification
    out = self.forward(x)
    loss = my_loss(out, y)
    return {'loss': loss} 
```

**And you also decide how to collate the output of all validation steps**    

```python
def validation_end(self, outputs):
    """
    Called at the end of validation to aggregate outputs
    :param outputs: list of individual outputs of each validation step
    :return:
    """
    val_loss_mean = 0
    val_acc_mean = 0
    for output in outputs:
        val_loss_mean += output['val_loss']
        val_acc_mean += output['val_acc']

    val_loss_mean /= len(outputs)
    val_acc_mean /= len(outputs)
    logs = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    result = {'log': logs}
    return result
```
   
## Tensorboard    
Lightning is fully integrated with tensorboard, MLFlow and supports any logging module.   

![tensorboard-support](docs/source/_static/images/tf_loss.png)

Lightning also adds a text column with all the hyperparameters for this experiment.      

![tensorboard-support](docs/source/_static/images/tf_tags.png)

## Lightning automates all of the following ([each is also configurable](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.html)):


- [Running grid search on a cluster](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.distrib_data_parallel.html)  
- [Fast dev run](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.utilities.debugging.html)
- [Logging](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.logging.html)
- [Implement Your Own Distributed (DDP) training](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_ddp)
- [Multi-GPU & Multi-node](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.distrib_parts.html)
- [Training loop](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.training_loop.html)
- [Hooks](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.core.hooks.html)
- [Configure optimizers](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers)
- [Validations](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.evaluation_loop.html)
- [Model saving & Restoring training session](https://pytorch-lightning.rtfd.io/en/latest/pytorch_lightning.trainer.training_io.html)
  

## Examples   
- [GAN](https://github.com/PytorchLightning/pytorch-lightning/tree/master/pl_examples/domain_templates/gan.py)    
- [MNIST](https://github.com/PytorchLightning/pytorch-lightning/tree/master/pl_examples/basic_examples)      
- [Other projects using Lightning](https://github.com/PytorchLightning/pytorch-lightning/network/dependents?package_id=UGFja2FnZS0zNzE3NDU4OTM%3D)    
- [Multi-node](https://github.com/PytorchLightning/pytorch-lightning/tree/master/pl_examples/multi_node_examples)   

## Tutorials   
- [Basic Lightning use](https://towardsdatascience.com/supercharge-your-ai-research-with-pytorch-lightning-337948a99eec)    
- [9 key speed features in Pytorch-Lightning](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565)    
- [SLURM, multi-node training with Lightning](https://towardsdatascience.com/trivial-multi-node-training-with-pytorch-lightning-ff75dfb809bd)     

---

## Asking for help    
Welcome to the Lightning community!   

If you have any questions, feel free to:   
1. [read the docs](https://pytorch-lightning.rtfd.io/en/latest/).     
2. [Search through the issues](https://github.com/PytorchLightning/pytorch-lightning/issues?utf8=%E2%9C%93&q=my++question).      
3. [Ask on stackoverflow](https://stackoverflow.com/questions/ask?guided=false) with the tag pytorch-lightning.   

If no one replies to you quickly enough, feel free to post the stackoverflow link to our Gitter chat!   

To chat with the rest of us visit our [gitter channel](https://gitter.im/PyTorch-Lightning/community)!     

---   
## FAQ    
**How do I use Lightning for rapid research?**   
[Here's a walk-through](https://pytorch-lightning.rtfd.io/en/latest/)  

**Why was Lightning created?**     
Lightning has 3 goals in mind:
1. Maximal flexibility while abstracting out the common boilerplate across research projects.   
2. Reproducibility. If all projects use the LightningModule template, it will be much much easier to understand what's going on and where to look! It will also mean every implementation follows a standard format.   
3. Democratizing PyTorch power user features. Distributed training? 16-bit? know you need them but don't want to take the time to implement? All good... these come built into Lightning.    

**How does Lightning compare with Ignite and fast.ai?**     
[Here's a thorough comparison](https://medium.com/@_willfalcon/pytorch-lightning-vs-pytorch-ignite-vs-fast-ai-61dc7480ad8a).    

**Is this another library I have to learn?**    
Nope! We use pure Pytorch everywhere and don't add unecessary abstractions!   

**Are there plans to support Python 2?**          
Nope.     

**Are there plans to support virtualenv?**    
Nope. Please use anaconda or miniconda.    

**Which PyTorch versions do you support?**    
- **PyTorch 1.1.0**       
    ```bash    
    # install pytorch 1.1.0 using the official instructions   
    
    # install test-tube 0.6.7.6 which supports 1.1.0   
    pip install test-tube==0.6.7.6   
    
    # install latest Lightning version without upgrading deps    
    pip install -U --no-deps pytorch-lightning
    ```     
- **PyTorch 1.2.0, 1.3.0,**
    Install via pip as normal   

## Custom installation

### Bleeding edge

If you can't wait for the next release, install the most up to date code with:
* using GIT (locally clone whole repo with full history)
    ```bash
    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
    ```
* using instant zip (last state of the repo without git history)
    ```bash
    pip install https://github.com/PytorchLightning/pytorch-lightning/archive/master.zip --upgrade
    ```

### Any release installation

You can also install any past release `0.X.Y` from this repository:
```bash
pip install https://github.com/PytorchLightning/pytorch-lightning/archive/0.X.Y.zip --upgrade
```

## Bibtex
If you want to cite the framework feel free to use this (but only if you loved it 😊):
```
@misc{Falcon2019,
  author = {Falcon, W.A. et al.},
  title = {PyTorch Lightning},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PytorchLightning/pytorch-lightning}}
}
```
