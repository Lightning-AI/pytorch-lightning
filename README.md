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
Keras and fast.ai are too abstract for researchers. Lightning abstracts the full training loop but gives you control in the critical points.   


## Why do I want to use lightning?
Because you want to use best practices and get gpu training, multi-node training, checkpointing, mixed-precision, etc... for free, but still want granular control of the meat of the training, validation and testing loops.

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

- 16 bit precision training (must have apex installed)  
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


## Demo
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


