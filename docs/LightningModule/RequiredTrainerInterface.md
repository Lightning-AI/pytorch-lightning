# Lightning Module interface
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/root_module.py)]

A lightning module is a strict superclass of nn.Module, it provides a standard interface for the trainer to interact with the model.

The easiest thing to do is copy [this template](../../examples/new_project_templates/lightning_module_template.py) and modify accordingly. 

Otherwise, to Define a Lightning Module, implement the following methods:

**Required**:  

- [training_step](RequiredTrainerInterface.md#training_step)   
- [validation_step](RequiredTrainerInterface.md#validation_step)
- [validation_end](RequiredTrainerInterface.md#validation_end)
    
- [configure_optimizers](RequiredTrainerInterface.md#configure_optimizers)
- [get_save_dict](RequiredTrainerInterface.md#get_save_dict)
- [load_model_specific](RequiredTrainerInterface.md#load_model_specific)

- [tng_dataloader](RequiredTrainerInterface.md#tng_dataloader)
- [tng_dataloader](RequiredTrainerInterface.md#tng_dataloader)
- [test_dataloader](RequiredTrainerInterface.md#test_dataloader)

**Optional**:   

- [update_tng_log_metrics](RequiredTrainerInterface.md#update_tng_log_metrics)
- [add_model_specific_args](RequiredTrainerInterface.md#add_model_specific_args)

---

### training_step

``` {.python}
def training_step(self, data_batch, batch_nb)
```

In this step you'd normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something specific to your model.

**Params**    

| Param  | description  |
|---|---|
|  data_batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |

**Return**   

Dictionary or OrderedDict   

| key  | value  | is required |
|---|---|---|
|  loss | tensor scalar  | Y |
|  prog | Dict for progress bar display. Must have only tensors  | N |


**Example**

``` {.python}
def training_step(self, data_batch, batch_nb):
    x, y, z = data_batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, x)
    
    output = {
        'loss': loss, # required
        'prog': {'tng_loss': loss, 'batch_nb': batch_nb} # optional
    }
    
    # return a dict
    return output
```

---

### validation_step

``` {.python}
def validation_step(self, data_batch, batch_nb)
```

In this step you'd normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something specific to your model.
This is most likely the same as your training_step. But unlike training step, the outputs from here will go to validation_end for collation.

**Params**   

| Param  | description  |
|---|---|
|  data_batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict of OrderedDict with metrics to display in progress bar. All keys must be tensors. | Y |

**Example**

``` {.python}
def validation_step(self, data_batch, batch_nb):
    x, y, z = data_batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, x)
    
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

--- 
### validation_end

``` {.python}
def validation_end(self, outputs)
```

Called at the end of the validation loop with the output of each validation_step.

**Params**    

| Param  | description  |
|---|---|
|  outputs | List of outputs you defined in validation_step |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict of OrderedDict with metrics to display in progress bar | Y |

**Example**

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
    tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    return tqdm_dic
```

--- 
### configure_optimizers 

``` {.python}
def configure_optimizers(self)
```

Set up as many optimizers as you need. Normally you'd need one. But in the case of GANs or something more esoteric you might have multiple. 
Lightning will call .backward() and .step() on each one.  If you use 16 bit precision it will also handle that.


##### Return
List - List of optimizers

**Example**

``` {.python}
# most cases
def configure_optimizers(self):
    opt = Adam(lr=0.01)
    return [opt]
    
# gan example
def configure_optimizers(self):
    generator_opt = Adam(lr=0.01)
    disriminator_opt = Adam(lr=0.02)
    return [generator_opt, disriminator_opt] 
```

--- 
### get_save_dict 

``` {.python}
def get_save_dict(self)
```
Called by lightning to checkpoint your model. Lightning saves current epoch, current batch nb, etc...
All you have to return is what specifically about your lightning model you want to checkpoint.

##### Return
Dictionary - No required keys. Most of the time as described in this example.   

**Example**

``` {.python}
def get_save_dict(self):
    # 99% of use cases this is all you need to return
    checkpoint = {'state_dict': self.state_dict()}
    return checkpoint
```

--- 
### load_model_specific 

``` {.python}
def load_model_specific(self, checkpoint)
```
Called by lightning to restore your model. This is your chance to restore your model using the keys you added in get_save_dict.
Lightning will automatically restore current epoch, batch nb, etc. 

##### Return
Nothing  

**Example**

``` {.python}
def load_model_specific(self, checkpoint):
    # you defined 'state_dict' in get_save_dict()
    self.load_state_dict(checkpoint['state_dict'])
```

--- 
### tng_dataloader 

``` {.python}
@property
def tng_dataloader(self)
```
Called by lightning during training loop. Define it as a property.

##### Return
Pytorch DataLoader

**Example**

``` {.python}
@property
def tng_dataloader(self):
    if self._tng_dataloader is None:
        try:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
            dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform, download=True)
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True
            )
            self._tng_dataloader = loader
        except Exception as e:
            raise e
            
    return self._tng_dataloader
```

--- 
### val_dataloader 

``` {.python}
@property
def tng_dataloader(self)
```
Called by lightning during validation loop. Define it as a property.

##### Return
Pytorch DataLoader

**Example**

``` {.python}
@property
def val_dataloader(self):
    if self._val_dataloader is None:
        try:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
            dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True
            )
            self._val_dataloader = loader
        except Exception as e:
            raise e
            
    return self._val_dataloader
```

--- 
### test_dataloader 

``` {.python}
@property
def test_dataloader(self)
```
Called by lightning during test loop. Define it as a property.

##### Return
Pytorch DataLoader

**Example**

``` {.python}
@property
def test_dataloader(self):
    if self._test_dataloader is None:
        try:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
            dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True
            )
            self._test_dataloader = loader
        except Exception as e:
            raise e
            
    return self._test_dataloader
```

--- 
### update_tng_log_metrics 

``` {.python}
def update_tng_log_metrics(self, logs)
```
Called by lightning right before it logs metrics for this batch.
This is a chance to ammend or add to the metrics about to be logged.

##### Return
Dict 

**Example**

``` {.python}
def update_tng_log_metrics(self, logs):
    # modify or add to logs
    return logs
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
    # parser.set_defaults(gradient_clip=5.0)

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