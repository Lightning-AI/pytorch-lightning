<p align="center">
  <a href="https://williamfalcon.github.io/pytorch-lightning/">
    <img alt="" src="https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/_static/lightning_logo.png" width="50">
  </a>
</p>
<h3 align="center">
  Pytorch Lightning
</h3>
<p align="center">
  The Keras for ML researchers using PyTorch. More control. Less boilerplate.    
</p>
<p align="center">
  <a href="https://badge.fury.io/py/pytorch-lightning"><img src="https://badge.fury.io/py/pytorch-lightning.svg" alt="PyPI version" height="18"></a>
<!--   <a href="https://travis-ci.org/williamFalcon/test-tube"><img src="https://travis-ci.org/williamFalcon/pytorch-lightning.svg?branch=master"></a> -->
  <a href="https://github.com/williamFalcon/pytorch-lightning/blob/master/COPYING"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>   

```bash
pip install pytorch-lightning    
```

## Docs   
**[View the docs here](https://williamfalcon.github.io/pytorch-lightning/)**

## What is it?  
Keras is too abstract for researchers. Lightning abstracts the full training loop but gives you control in the critical points.   


## Why do I want to use lightning?
Because you want to use best practices and get gpu training, multi-node training, checkpointing, mixed-precision, etc... for free.

To use lightning do 2 things:  
1. [Define a trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/examples/basic_trainer.py) (which will run ALL your models).   
2. [Define a model](https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/examples/example_model.py).     

## What are some key lightning features?

- Automatic training loop
```python
# define what happens for training here
def training_step(self, data_batch, batch_nb):
    x, y = data_batch
    out = self.forward(x)
    loss = my_loss(out, y)
    return {'loss': loss} 
```
- Automatic validation loop   

```python
# define what happens for validation here
def validation_step(self, data_batch, batch_nb):    x, y = data_batch
    out = self.forward(x)
    loss = my_loss(out, y)
    return {'loss': loss} 
```

- Automatic early stopping   
```python
callback = EarlyStopping(...)
Trainer(early_stopping=callback)
```

- Learning rate annealing     
```python
# anneal at 100 and 200 epochs
Trainer(lr_scheduler_milestones=[100, 200])
```

- 16 bit precision training   
```python
Trainer(use_amp=True, amp_level='O2')
```

- multi-gpu training
```python
# train on 4 gpus
Trainer(gpus=[0, 1, 2, 3])
```

- Automatic checkpointing
```python
# do 3 things:
# 1
Trainer(checkpoint_callback=ModelCheckpoint)

# 2 return what to save in a checkpoint
def get_save_dict(self):
    return {'state_dict': self.state_dict()}


# 3 use the checkpoint to reset your model state
def load_model_specific(self, checkpoint):
    self.load_state_dict(checkpoint['state_dict'])
```

- Log all details of your experiment (model params, code snapshot, etc...)
```python
from test_tube import Experiment

exp = Experiment(...)
Trainer(experiment=exp)
```

- Run grid-search on cluster
```python
from test_tube import Experiment, SlurmCluster, HyperOptArgumentParser

def training_fx(hparams, cluster, _):
    # hparams are local params
    model = MyModel()
    trainer = Trainer(...)
    trainer.fit(model)

# grid search number of layers
parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--layers', default=5, type=int, options=[1, 5, 10, 20, 50])
hyperparams = parser.parse_args()

cluster = SlurmCluster(hyperparam_optimizer=hyperparams)
cluster.optimize_parallel_cluster_gpu(training_fx)
```


#### Quick demo
Run the following demo to see how it works:
```bash
# install lightning
pip install pytorch-lightning

# clone lightning for the demo
git clone https://github.com/williamFalcon/pytorch-lightning.git
cd pytorch-lightning/docs/source/examples

# run demo (on cpu)
python fully_featured_trainer.py
```

Without changing the model AT ALL, you can run the model on a single gpu, over multiple gpus, or over multiple nodes.
```bash
# run a grid search on two gpus
python fully_featured_trainer.py --gpus "0;1"

# run single model on multiple gpus
python fully_featured_trainer.py --gpus "0;1" --interactive
```

#### Basic trainer example  
See [this demo](https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/examples/fully_featured_trainer.py) for a more robust trainer example.   

```python
import os
import sys

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.utils.arg_parse import add_default_args
from pytorch_lightning.utils.pt_callbacks import EarlyStopping, ModelCheckpoint
from demo.example_model import ExampleModel


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment
    exp = Experiment(
        name=hparams.tt_name,
        debug=hparams.debug,
        save_dir=hparams.tt_save_path,
        version=hparams.hpc_exp_number,
        autosave=False,
        description=hparams.tt_description
    )

    exp.argparse(hparams)
    exp.save()

    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)

    # build model
    model = ExampleModel(hparams)

    # callbacks
    early_stop = EarlyStopping(monitor='val_acc', patience=3, mode='min', verbose=True)
    checkpoint = ModelCheckpoint(filepath=model_save_path, save_function=None, save_best_only=True, verbose=True, monitor='val_acc', mode='min')

    # configure trainer
    trainer = Trainer(experiment=exp, checkpoint_callback=checkpoint, early_stop_callback=early_stop)

    # train model
    trainer.fit(model)


if __name__ == '__main__':

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
    add_default_args(parent_parser, root_dir)

    # allow model to overwrite or extend args
    parser = ExampleModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)

```

#### Basic model example  
Here we only show the method signatures. It's up to you to define the content.   

```python
from torch import nn

class My_Model(RootModule):
    def __init__(self):
        # define model
        self.l1 = nn.Linear(200, 10)

    # ---------------
    # TRAINING
    def training_step(self, data_batch):
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)

        return loss_val, {'train_loss': loss}

    def validation_step(self, data_batch):
        x, y = data_batch
        y_hat = self.l1(x)
        loss = some_loss(y_hat)

        return loss_val, {'val_loss': loss}

     def validation_end(self, outputs):
        total_accs = []

        for output in outputs:
            total_accs.append(output['val_acc'].item())

        # return a dict
        return {'total_acc': np.mean(total_accs)}

     # ---------------
     # SAVING
     def get_save_dict(self):
        # lightning saves for you. Here's your chance to say what you want to save
        checkpoint = {'state_dict': self.state_dict()}

        return checkpoint

    def load_model_specific(self, checkpoint):
        # lightning loads for you. Here's your chance to say what you want to load
        self.load_state_dict(checkpoint['state_dict'])

    # ---------------
    # TRAINING CONFIG
    def configure_optimizers(self):
        # give lightning the list of optimizers you want to use.
        # lightning will call automatically
        optimizer = self.choose_optimizer('adam', self.parameters(), {'lr': self.hparams.learning_rate}, 'optimizer')
        return [optimizer]

    @property
    def tng_dataloader(self):
        return pytorch_dataloader('train')

    @property
    def val_dataloader(self):
        return pytorch_dataloader('val')

    @property
    def test_dataloader(self):
        return pytorch_dataloader('test')

    # ---------------
    # MODIFY YOUR COMMAND LINE ARGS
    @staticmethod
    def add_model_specific_args(parent_parser):    
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
        parser.add_argument('--out_features', default=20)
        return parser
```


### Details    

#### Model definition   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| training_step  |  Called with a batch of data during training | data from your dataloaders  | tuple: scalar, dict  |
| validation_step  |  Called with a batch of data during validation  | data from your dataloaders  | tuple: scalar, dict |
| validation_end  |  Collate metrics from all validation steps |  outputs: array where each item is the output of a validation step | dict: for logging |
| get_save_dict  | called when your model needs to be saved (checkpoints, hpc save, etc...)  | None  |  dict to be saved |

#### Model training   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| configure_optimizers  |  called during training setup | None | list: optimizers you want to use |
| tng_dataloader  |  called during training  | None  | pytorch dataloader |
| val_dataloader  |  called during validation  | None  | pytorch dataloader |
| test_dataloader  |  called during testing  | None  | pytorch dataloader |    
| add_model_specific_args  |  called with args you defined in your main. This lets you tailor args for each model and keep main the same  | argparse  | argparse |

#### Model Saving/Loading   
| Name  | Description  |  Input |  Return |
|---|---|---|---|
| get_save_dict  | called when your model needs to be saved (checkpoints, hpc save, etc...)  | None  |  dict to be saved |
|  load_model_specific |  called when loading a model | checkpoint: dict you created in get_save_dict  | dict: modified in whatever way you want  |

## Optional model hooks.   
Add these to the model whenever you want to configure training behavior.    


### Model lifecycle hooks
Use these hooks to customize functionality

| Method | Purpose  | Input  | Output  | Required  |
|---|---|---|---|---|
| on_batch_start()  | called right before the batch starts | - | -  | N  |
| on_batch_end()  | called right after the batch ends | - | -  | N  |
| on_epoch_start()  | called right before the epoch starts | - | -  | N  |
| on_epoch_end()  | called right afger the epoch ends | - | -  | N  |
| on_pre_performance_check()  | called right before the performance check starts | - | -  | N  |
| on_post_performance_check()  | called right after the batch starts | - | -  | N  |
