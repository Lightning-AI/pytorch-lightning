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
Because you don't want to define a training loop, validation loop, gradient clipping, checkpointing, loading, 
gpu training, etc... every time you start a project. Let lightning handle all of that for you and you just define your 
data and what happens in the training, testing and validation loop 

To use lightning do 2 things:  
1. [Define a Trainer](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/trainer_cpu_template.py).   
2. [Define a LightningModel](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/lightning_module_template.py).     

## What does lightning control for me?
Everything! Except the following three things:

**Automatic training loop**    

```python
# define what happens for training here
def training_step(self, data_batch, batch_nb):
    x, y = data_batch
    
    # define your own forward and loss calculation
    out = self.forward(x)
    loss = my_loss(out, y)
    return {'loss': loss} 
```

**Automatic validation loop**      

```python
# define what happens for validation here
def validation_step(self, data_batch, batch_nb):    
    x, y = data_batch
    
    # define your own forward and loss calculation
    out = self.forward(x)
    loss = my_loss(out, y)
    return {'loss': loss} 
```

**Collate the output of the validation_step**    

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
    tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    return tqdm_dic
```

## Lightning gives you options to control the following:

**Checkpointing**    

- Model saving
- Model loading 

**Computing cluster (SLURM)**    

- Automatic checkpointing   
- Automatic saving, loading  
- Running grid search on a cluster 
- Walltime auto-resubmit   

**Debugging**  

- [Fast dev run](Debugging/#fast-dev-run)
- [Inspect gradient norms](Debugging/#inspect-gradient-norms)
- [Log GPU usage](Debugging/#Log-gpu-usage)
- [Make model overfit on subset of data](Debugging/#make-model-overfit-on-subset-of-data)
- [Print the parameter count by layer](Debugging/#print-the-parameter-count-by-layer)
- [Pring which gradients are nan](Debugging/#print-which-gradients-are-nan)


**Distributed training**    

- [16-bit mixed precision](Distributed%20training/#16-bit-mixed-precision)
- [Multi-GPU](Distributed%20training/#Multi-GPU)
- [Multi-node](Distributed%20training/#Multi-node)
- [Single GPU](Distributed%20training/#single-gpu)
- [Self-balancing architecture](Distributed%20training/#self-balancing-architecture)


**Experiment Logging**   

- [Display metrics in progress bar](Logging/#display-metrics-in-progress-bar)
- Log arbitrary metrics
- [Log metric row every k batches](Logging/#log-metric-row-every-k-batches)
- [Process position](Logging/#process-position)
- [Save a snapshot of all hyperparameters](Logging/#save-a-snapshot-of-all-hyperparameters) 
- [Snapshot code for a training run](Logging/#snapshot-code-for-a-training-run) 
- [Write logs file to csv every k batches](Logging/#write-logs-file-to-csv-every-k-batches)

**Training loop**    

- [Accumulate gradients](Training%20Loop/#accumulated-gradients)
- [Anneal Learning rate](Training%20Loop/#anneal-learning-rate)
- [Force training for min or max epochs](Training%20Loop/#force-training-for-min-or-max-epochs)
- [Force disable early stop](Training%20Loop/#force-disable-early-stop)
- [Use multiple optimizers (like GANs)](../Pytorch-lightning/LightningModule/#configure_optimizers)
- [Set how much of the training set to check (1-100%)](Training%20Loop/#set-how-much-of-the-training-set-to-check)

**Validation loop**    

- [Check validation every n epochs](Validation%20Loop/#check-validation-every-n-epochs)
- [Set how much of the validation set to check](Validation%20Loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](Validation%20Loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](Validation%20Loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](Validation%20Loop/#set-the-number-of-validation-sanity-steps)


## Demo
```bash
# install lightning
pip install pytorch-lightning

# clone lightning for the demo
git clone https://github.com/williamFalcon/pytorch-lightning.git
cd examples/new_project_templates/

# run demo (on cpu)
python trainer_gpu_cluster_template.py
```

Without changing the model AT ALL, you can run the model on a single gpu, over multiple gpus, or over multiple nodes.
```bash
# run a grid search on two gpus
python fully_featured_trainer.py --gpus "0;1"

# run single model on multiple gpus
python fully_featured_trainer.py --gpus "0;1" --interactive
```


