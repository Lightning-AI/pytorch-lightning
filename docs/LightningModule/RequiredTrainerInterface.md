# Lightning Module interface
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/root_module.py)]

A lightning module is a strict superclass of nn.Module, it provides a standard interface for the trainer to interact with the model.

The easiest thing to do is copy the [minimal example](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#minimal-example) below and modify accordingly. 

Otherwise, to Define a Lightning Module, implement the following methods:

**Required**:  

- [training_step](RequiredTrainerInterface.md#training_step)      
- [train_dataloader](RequiredTrainerInterface.md#train_dataloader)    
- [configure_optimizers](RequiredTrainerInterface.md#configure_optimizers)    

**Optional**:   

- [validation_step](RequiredTrainerInterface.md#validation_step)    
- [validation_end](RequiredTrainerInterface.md#validation_end) 
- [test_step](RequiredTrainerInterface.md#test_step)    
- [test_end](RequiredTrainerInterface.md#test_end) 
- [val_dataloader](RequiredTrainerInterface.md#val_dataloader)    
- [test_dataloader](RequiredTrainerInterface.md#test_dataloader)    
- [on_save_checkpoint](RequiredTrainerInterface.md#on_save_checkpoint)    
- [on_load_checkpoint](RequiredTrainerInterface.md#on_load_checkpoint)    
- [add_model_specific_args](RequiredTrainerInterface.md#add_model_specific_args)    

---
### Minimal example   
```python
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import pytorch_lightning as pl

class CoolModel(pl.LightningModule):

    def __init__(self):
        super(CoolModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
```
---   
### How do these methods fit into the broader training?     
The LightningModule interface is on the right. Each method corresponds to a part of a research project. Lightning automates everything not in blue.    

<p align="center">
  <a href="https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/_static/overview_flat.jpg">
    <img alt="" src="https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/_static/overview_flat.jpg" height="900px">
  </a>
</p>   

## Required Methods    

### training_step

``` {.python}
def training_step(self, batch, batch_nb)
```

In this step you'd normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something specific to your model.

**Params**    

| Param  | description  |
|---|---|
|  batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |

**Return**   

Dictionary or OrderedDict   

| key  | value  | is required |
|---|---|---|
|  loss | tensor scalar  | Y |
|  progress_bar | Dict for progress bar display. Must have only tensors  | N |
|  log | Dict of metrics to add to logger. Must have only tensors (no images, etc)  | N |


**Example**

``` {.python}
def training_step(self, batch, batch_nb):
    x, y, z = batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, x)
    
    output = {
        'loss': loss, # required
        'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
        'log': {'training_loss': loss} # optional (MUST ALL BE TENSORS)
    }
    
    # return a dict
    return output
```    

If you define multiple optimizers, this step will also be called with an additional ```optimizer_idx``` param.    
``` {.python}
# Multiple optimizers (ie: GANs)     
def training_step(self, batch, batch_nb, optimizer_idx):
    if optimizer_idx == 0:
        # do training_step with encoder
    if optimizer_idx == 1:
        # do training_step with decoder    
```    

You can also return a -1 instead of a dict to stop the current loop. This is useful if you want to
break out of the current training epoch early.

--- 
### train_dataloader 

``` {.python}
@pl.data_loader
def train_dataloader(self)
```
Called by lightning during training loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.   
If you want to change the data during every epoch DON'T use the data_loader decorator.   

##### Return
PyTorch DataLoader

**Example**

``` {.python}
@pl.data_loader
def train_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    return loader
```

--- 
### configure_optimizers 

``` {.python}
def configure_optimizers(self)
```

Set up as many optimizers and (optionally) learning rate schedulers as you need. Normally you'd need one. But in the case of GANs or something more esoteric you might have multiple. 
Lightning will call .backward() and .step() on each one in every epoch.  If you use 16 bit precision it will also handle that.

**Note:** If you use multiple optimizers, training_step will have an additional ```optimizer_idx``` parameter.    
**Note 2:** If you use LBFGS lightning handles the closure function automatically for you.

##### Return    
Return any of these 3 options:   
Single optimizer   
List or Tuple - List of optimizers    
Two lists - The first list has multiple optimizers, the second a list of learning-rate schedulers

**Example**

``` {.python}
# most cases
def configure_optimizers(self):
    opt = Adam(self.parameters(), lr=0.01)
    return opt

# multiple optimizer case (eg: GAN)
def configure_optimizers(self):
    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
    return generator_opt, disriminator_opt
    
# example with learning_rate schedulers  
def configure_optimizers(self):
    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
    return [generator_opt, disriminator_opt], [discriminator_sched]
```  

If you need to control how often those optimizers step or override the default .step() schedule, override 
the [optimizer_step](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/#optimizer_step) hook.   

## Optional Methods    

### validation_step

``` {.python}
# if you have one val dataloader:
def validation_step(self, batch, batch_nb)   

# if you have multiple val dataloaders:  
def validation_step(self, batch, batch_nb, dataloader_idxdx)
```
**OPTIONAL**    
If you don't need to validate you don't need to implement this method. In this step you'd normally generate examples or calculate anything of interest such as accuracy. 

When the validation_step is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.

The dict you return here will be available in the `validation_end` method. 

**Params**   

| Param  | description  |
|---|---|
|  batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |
|  dataloader_idx | Integer displaying which dataloader this is (only if multiple val datasets used)  |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict or OrderedDict - passed to the validation_end step | N |

**Example**

``` {.python}
# CASE 1: A single validation dataset
def validation_step(self, batch, batch_nb):
    x, y = batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, y)
    
    # log 6 example images
    # or generated text... or whatever
    sample_imgs = x[:6]
    grid = torchvision.utils.make_grid(sample_imgs)
    self.logger.experiment.add_image('example_images', grid, 0) 
    
    # calculate acc
    labels_hat = torch.argmax(out, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    
    # all optional...
    # return whatever you need for the collation function validation_end
    output = OrderedDict({
        'val_loss': loss_val,
        'val_acc': torch.tensor(val_acc), # everything must be a tensor
    })
    
    # return an optional dict
    return output
```   

If you pass in multiple validation datasets, validation_step will have an additional argument.

```python
# CASE 2: multiple validation datasets
def validation_step(self, batch, batch_nb, dataset_idx):
    # dataset_idx tells you which dataset this is.   
```   

The ```dataset_idx``` corresponds to the order of datasets returned in ```val_dataloader```.    

--- 
### validation_end

``` {.python}
def validation_end(self, outputs)
```   
If you didn't define a validation_step, this won't be called.       

Called at the end of the validation loop with the outputs of validation_step.

The outputs here are strictly for the progress bar. If you don't need to display anything, don't return anything.    

**Params**    

| Param  | description  |
|---|---|
|  outputs | List of outputs you defined in validation_step, or if there are multiple dataloaders, a list containing a list of outputs for each dataloader |

**Return**   

Dictionary or OrderedDict   

| key  | value  | is required |
|---|---|---|
|  progress_bar | Dict for progress bar display. Must have only tensors  | N |
|  log | Dict of metrics to add to logger. Must have only tensors (no images, etc)  | N |

**Example**

With a single dataloader

``` {.python}
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
    tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
       
    # show val_loss and val_acc in progress bar but only log val_loss
    results = {
        'progress_bar': tqdm_dict,
        'log': {'val_loss': val_loss_mean.item()}
    }
    return results
```

With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
one entry per dataloader, while the inner list contains the individual outputs of 
each validation step for that dataloader.

``` {.python}
def validation_end(self, outputs):
    """
    Called at the end of validation to aggregate outputs
    :param outputs: list of list of individual outputs of each validation step
    :return:
    """
    val_loss_mean = 0
    val_acc_mean = 0
    i = 0
    for dataloader_outputs in outputs:
        for output in dataloader_outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
            i += 1

    val_loss_mean /= i
    val_acc_mean /= i
    tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    
    # show val_loss and val_acc in progress bar but only log val_loss
    results = {
        'progress_bar': tqdm_dict,
        'log': {'val_loss': val_loss_mean.item()}
    }
    return results
```

### test_step

``` {.python}
# if you have one test dataloader:
def test_step(self, batch, batch_nb)   

# if you have multiple test dataloaders:  
def test_step(self, batch, batch_nb, dataloader_idxdx)
```
**OPTIONAL**    
If you don't need to test you don't need to implement this method. In this step you'd normally generate examples or calculate anything of interest such as accuracy.   

When the validation_step is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of validation, model goes back to training mode and gradients are enabled.

The dict you return here will be available in the `test_end` method. 

This function is used when you execute `trainer.test()`.

**Params**   

| Param  | description  |
|---|---|
|  batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |
|  dataloader_idx | Integer displaying which dataloader this is (only if multiple test datasets used)  |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict or OrderedDict with metrics to display in progress bar. All keys must be tensors. | Y |

**Example**

``` {.python}
# CASE 1: A single test dataset
def test_step(self, batch, batch_nb):
    x, y = batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, y)
    
    # calculate acc
    labels_hat = torch.argmax(out, dim=1)
    test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    
    # all optional...
    # return whatever you need for the collation function test_end
    output = OrderedDict({
        'test_loss': loss_test,
        'test_acc': torch.tensor(test_acc), # everything must be a tensor
    })
    
    # return an optional dict
    return output
```   

If you pass in multiple test datasets, test_step will have an additional argument.

```python
# CASE 2: multiple test datasets
def test_step(self, batch, batch_nb, dataset_idx):
    # dataset_idx tells you which dataset this is.   
```   

The ```dataset_idx``` corresponds to the order of datasets returned in ```test_dataloader```.    

--- 
### test_end

``` {.python}
def test_end(self, outputs)
```   
If you didn't define a test_step, this won't be called.       

Called at the end of the test step with the output of each test_step.

The outputs here are strictly for the progress bar. If you don't need to display anything, don't return anything.    

**Params**    

| Param  | description  |
|---|---|
|  outputs | List of outputs you defined in test_step, or if there are multiple dataloaders, a list containing a list of outputs for each dataloader |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict of OrderedDict with metrics to display in progress bar | Y |

**Example**

``` {.python}
def test_end(self, outputs):
    """
    Called at the end of test to aggregate outputs
    :param outputs: list of individual outputs of each test step
    :return:
    """
    test_loss_mean = 0
    test_acc_mean = 0
    for output in outputs:
        test_loss_mean += output['test_loss']
        test_acc_mean += output['test_acc']

    test_loss_mean /= len(outputs)
    test_acc_mean /= len(outputs)
    tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
    
    # show test_loss and test_acc in progress bar but only log test_loss
    results = {
        'progress_bar': tqdm_dict,
        'log': {'test_loss': val_loss_mean.item()}
    }
    return results
```

With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
one entry per dataloader, while the inner list contains the individual outputs of 
each validation step for that dataloader.

``` {.python}
def test_end(self, outputs):
    """
    Called at the end of test to aggregate outputs
    :param outputs: list of individual outputs of each test step
    :return:
    """
    test_loss_mean = 0
    test_acc_mean = 0
    i = 0
    for dataloader_outputs in outputs:
        for output in dataloader_outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']
            i += 1

    test_loss_mean /= i 
    test_acc_mean /= i
    tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
    
    # show test_loss and test_acc in progress bar but only log test_loss
    results = {
        'progress_bar': tqdm_dict,
        'log': {'test_loss': val_loss_mean.item()}
    }
    return results
```

--- 
### on_save_checkpoint 

``` {.python}
def on_save_checkpoint(self, checkpoint)
```
Called by lightning to checkpoint your model. Lightning saves the training state (current epoch, global_step, etc)
and also saves the model state_dict. If you want to save anything else, use this method to add your own
key-value pair.

##### Return
Nothing

**Example**

``` {.python}
def on_save_checkpoint(self, checkpoint):
    # 99% of use cases you don't need to implement this method 
    checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
```

--- 
### on_load_checkpoint 

``` {.python}
def on_load_checkpoint(self, checkpoint)
```
Called by lightning to restore your model. Lighting auto-restores global step, epoch, etc...
It also restores the model state_dict.
If you saved something with **on_save_checkpoint** this is your chance to restore this.

##### Return
Nothing  

**Example**

``` {.python}
def on_load_checkpoint(self, checkpoint):
    # 99% of the time you don't need to implement this method
    self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']
```

--- 
### val_dataloader 

``` {.python}
@pl.data_loader
def val_dataloader(self)
```
**OPTIONAL**    
If you don't need a validation dataset and a validation_step, you don't need to implement this method.    

Called by lightning during validation loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.   
If you want to change the data during every epoch DON'T use the data_loader decorator.   

##### Return
PyTorch DataLoader or list of PyTorch Dataloaders.    

**Example**

``` {.python}
@pl.data_loader
def val_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    
    return loader

# can also return multiple dataloaders   
@pl.data_loader
def val_dataloader(self):
    return [loader_a, loader_b, ..., loader_n]   
```

In the case where you return multiple val_dataloaders, the validation_step will have an arguement ```dataset_idx```
which matches the order here.    

--- 
### test_dataloader 

``` {.python}
@pl.data_loader
def test_dataloader(self)
```
**OPTIONAL**    
If you don't need a test dataset and a test_step, you don't need to implement this method.    

Called by lightning during test loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.
If you want to change the data during every epoch DON'T use the data_loader decorator.   

##### Return
PyTorch DataLoader

**Example**

``` {.python}
@pl.data_loader
def test_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    
    return loader
```

--- 
### add_model_specific_args 

``` {.python}
@staticmethod
def add_model_specific_args(parent_parser, root_dir)
```
Lightning has a list of default argparse commands.
This method is your chance to add or modify commands specific to your model.
The [hyperparameter argument parser](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/) is available anywhere in your model by calling self.hparams.

##### Return
An argument parser

**Example**

``` {.python}
@staticmethod
def add_model_specific_args(parent_parser, root_dir):
    parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

    # param overwrites
    # parser.set_defaults(gradient_clip_val=5.0)

    # network params
    parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)
    parser.add_argument('--in_features', default=28*28)
    parser.add_argument('--out_features', default=10)
    parser.add_argument('--hidden_dim', default=50000) # use 500 for CPU, 50000 for GPU to see speed difference

    # data
    parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

    # training params (opt)
    parser.opt_list('--learning_rate', default=0.001, type=float, options=[0.0001, 0.0005, 0.001, 0.005],
                    tunable=False)
    parser.opt_list('--batch_size', default=256, type=int, options=[32, 64, 128, 256], tunable=False)
    parser.opt_list('--optimizer_name', default='adam', type=str, options=['adam'], tunable=False)
    return parser
```
