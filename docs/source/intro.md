## New project Quick Start    
To start a new project define two files, a LightningModule and a Trainer file.    
To illustrate Lightning power and simplicity, here's an example of a typical research flow.    

### Case 1: BERT    
Let's say you're working on something like BERT but want to try different ways of training or even different networks.  
You would define a single LightningModule and use flags to switch between your different ideas.   
```python
class BERT(pl.LightningModule):
    def __init__(self, model_name, task):
        self.task = task
    
        if model_name == 'transformer':
            self.net = Transformer()
        elif model_name == 'my_cool_version':
            self.net = MyCoolVersion()
            
    def training_step(self, batch, batch_nb):
        if self.task == 'standard_bert':
            # do standard bert training with self.net...
            # return loss
            
        if self.task == 'my_cool_task':
            # do my own version with self.net
            # return loss
```   

### Case 2: COOLER NOT BERT    
But if you wanted to try something **completely** different, you'd define a new module for that.    
```python

class CoolerNotBERT(pl.LightningModule):
    def __init__(self):
        self.net = ...
        
    def training_step(self, batch, batch_nb):
        # do some other cool task
        # return loss   
```   

### Rapid research flow    
Then you could do rapid research by switching between these two and using the same trainer.   
```python

if use_bert:
    model = BERT()
else:
    model = CoolerNotBERT()
    
trainer = Trainer(gpus=4, use_amp=True)
trainer.fit(model)
```

Notice a few things about this flow:   
1. You're writing pure PyTorch... no unnecessary abstractions or new libraries to learn.   
2. You get free GPU and 16-bit support without writing any of that code in your model.   
3. You also get all of the capabilities below (without coding or testing yourself).     

---    
### Templates 
1. [MNIST LightningModule](LightningModule/RequiredTrainerInterface/#minimal-example) 
2. [Trainer](Trainer/)
    - [Basic CPU, GPU Trainer Template](https://github.com/williamFalcon/pytorch-lightning/tree/master/pl_examples/basic_examples)
    - [GPU cluster Trainer Template](https://github.com/williamFalcon/pytorch-lightning/tree/master/pl_examples/multi_node_examples)

### Docs shortcuts
- [LightningModule](../docs/source/LightningModule/RequiredTrainerInterface/)  
- [Trainer](../docs/source/Trainer/)  

### Quick start examples 
- [CPU example](../docs/source/Examples/Examples/#cpu-hyperparameter-search)   
- [Hyperparameter search on single GPU](../docs/source/Examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)    
- [Hyperparameter search on multiple GPUs on same node](../docs/source/Examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)  
- [Hyperparameter search on a SLURM HPC cluster](../docs/source/Examples/Examples/#Hyperparameter search on a SLURM HPC cluster)      


### Checkpointing    

- [Checkpoint callback](Trainer/Checkpointing/#model-saving)    
- [Model saving](Trainer/Checkpointing/#model-saving)
- [Model loading](LightningModule/methods/#load-from-metrics) 
- [Restoring training session](Trainer/Checkpointing/#restoring-training-session)

### Computing cluster (SLURM)    

- [Running grid search on a cluster](Trainer/SLURM%20Managed%20Cluster#running-grid-search-on-a-cluster) 
- [Walltime auto-resubmit](Trainer/SLURM%20Managed%20Cluster#walltime-auto-resubmit)   

### Debugging  

- [Fast dev run](Trainer/debugging/#fast-dev-run)
- [Inspect gradient norms](Trainer/debugging/#inspect-gradient-norms)
- [Log GPU usage](Trainer/debugging/#Log-gpu-usage)
- [Make model overfit on subset of data](Trainer/debugging/#make-model-overfit-on-subset-of-data)
- [Print the parameter count by layer](Trainer/debugging/#print-the-parameter-count-by-layer)
- [Pring which gradients are nan](Trainer/debugging/#print-which-gradients-are-nan)
- [Print input and output size of every module in system](LightningModule/properties/#example_input_array)


### Distributed training    

- [Implement Your Own Distributed (DDP) training](Trainer/hooks/#init_ddp_connection)
- [16-bit mixed precision](Trainer/Distributed%20training/#16-bit-mixed-precision)
- [Multi-GPU](Trainer/Distributed%20training/#Multi-GPU)
- [Multi-node](Trainer/Distributed%20training/#Multi-node)
- [Single GPU](Trainer/Distributed%20training/#single-gpu)
- [Self-balancing architecture](Trainer/Distributed%20training/#self-balancing-architecture)


### Experiment Logging   

- [Display metrics in progress bar](Trainer/Logging/#display-metrics-in-progress-bar)
- [Log metric row every k batches](Trainer/Logging/#log-metric-row-every-k-batches)
- [Process position](Trainer/Logging/#process-position)
- [Tensorboard support](Trainer/Logging/#tensorboard-support)
- [Save a snapshot of all hyperparameters](Trainer/Logging/#save-a-snapshot-of-all-hyperparameters) 
- [Snapshot code for a training run](Trainer/Logging/#snapshot-code-for-a-training-run) 
- [Write logs file to csv every k batches](Trainer/Logging/#write-logs-file-to-csv-every-k-batches)

### Training loop    

- [Accumulate gradients](Trainer/Training%20Loop/#accumulated-gradients)
- [Force training for min or max epochs](Trainer/Training%20Loop/#force-training-for-min-or-max-epochs)
- [Early stopping callback](Trainer/Training%20Loop/#early-stopping)    
- [Force disable early stop](Trainer/Training%20Loop/#force-disable-early-stop)
- [Gradient Clipping](Trainer/Training%20Loop/#gradient-clipping)
- [Hooks](Trainer/hooks/)
- [Learning rate scheduling](LightningModule/RequiredTrainerInterface/#configure_optimizers)
- [Use multiple optimizers (like GANs)](LightningModule/RequiredTrainerInterface/#configure_optimizers)
- [Set how much of the training set to check (1-100%)](Trainer/Training%20Loop/#set-how-much-of-the-training-set-to-check)
- [Step optimizers at arbitrary intervals](Trainer/hooks/#optimizer_step)

### Validation loop    

- [Check validation every n epochs](Trainer/Validation%20loop/#check-validation-every-n-epochs)
- [Hooks](Trainer/hooks/)
- [Set how much of the validation set to check](Trainer/Validation%20loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](Trainer/Validation%20loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](Trainer/Validation%20loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](Trainer/Validation%20loop/#set-the-number-of-validation-sanity-steps)

### Testing loop  
- [Run test set](Trainer/Testing%20loop/)  
