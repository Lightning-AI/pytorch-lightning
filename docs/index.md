###### New project Quick Start    
To start a new project define two files, a LightningModule and a Trainer file.    
To illustrate Lightning power and simplicity, here's an example of a typical research flow.    

###### Case 1: BERT    
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

###### Case 2: COOLER NOT BERT    
But if you wanted to try something **completely** different, you'd define a new module for that.    
```python

class CoolerNotBERT(pl.LightningModule):
    def __init__(self):
        self.net = ...
        
    def training_step(self, batch, batch_nb):
        # do some other cool task
        # return loss   
```   

###### Rapid research flow    
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
###### Templates 
1. [MNIST LightningModule](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#minimal-example) 
2. [Trainer](https://williamfalcon.github.io/pytorch-lightning/Trainer/)
    - [Basic CPU, GPU Trainer Template](https://github.com/williamFalcon/pytorch-lightning/tree/master/examples/basic_examples) 
    - [GPU cluster Trainer Template](https://github.com/williamFalcon/pytorch-lightning/tree/master/examples/multi_node_examples)

###### Docs shortcuts
- [LightningModule](LightningModule/RequiredTrainerInterface/)  
- [Trainer](Trainer/)  

###### Quick start examples 
- [CPU example](examples/Examples/#cpu-hyperparameter-search)   
- [Hyperparameter search on single GPU](examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)    
- [Hyperparameter search on multiple GPUs on same node](examples/Examples/#hyperparameter-search-on-a-single-or-multiple-gpus)  
- [Hyperparameter search on a SLURM HPC cluster](examples/Examples/#Hyperparameter search on a SLURM HPC cluster)      


###### Checkpointing    

- [Checkpoint callback](https://williamfalcon.github.io/pytorch-lightning/Trainer/Checkpointing/#model-saving)    
- [Model saving](https://williamfalcon.github.io/pytorch-lightning/Trainer/Checkpointing/#model-saving)
- [Model loading](https://williamfalcon.github.io/pytorch-lightning/LightningModule/methods/#load-from-metrics) 
- [Restoring training session](https://williamfalcon.github.io/pytorch-lightning/Trainer/Checkpointing/#restoring-training-session)

###### Computing cluster (SLURM)    

- [Running grid search on a cluster](https://williamfalcon.github.io/pytorch-lightning/Trainer/SLURM%20Managed%20Cluster#running-grid-search-on-a-cluster) 
- [Walltime auto-resubmit](https://williamfalcon.github.io/pytorch-lightning/Trainer/SLURM%20Managed%20Cluster#walltime-auto-resubmit)   

###### Debugging  

- [Fast dev run](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#fast-dev-run)
- [Inspect gradient norms](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#inspect-gradient-norms)
- [Log GPU usage](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#Log-gpu-usage)
- [Make model overfit on subset of data](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#make-model-overfit-on-subset-of-data)
- [Print the parameter count by layer](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#print-the-parameter-count-by-layer)
- [Pring which gradients are nan](https://williamfalcon.github.io/pytorch-lightning/Trainer/debugging/#print-which-gradients-are-nan)
- [Print input and output size of every module in system](https://williamfalcon.github.io/pytorch-lightning/LightningModule/properties/#example_input_array)


###### Distributed training    

- [16-bit mixed precision](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#16-bit-mixed-precision)
- [Multi-GPU](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#Multi-GPU)
- [Multi-node](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#Multi-node)
- [Single GPU](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#single-gpu)
- [Self-balancing architecture](https://williamfalcon.github.io/pytorch-lightning/Trainer/Distributed%20training/#self-balancing-architecture)


###### Experiment Logging   

- [Display metrics in progress bar](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#display-metrics-in-progress-bar)
- [Log metric row every k batches](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#log-metric-row-every-k-batches)
- [Process position](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#process-position)
- [Tensorboard support](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#tensorboard-support)
- [Save a snapshot of all hyperparameters](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#save-a-snapshot-of-all-hyperparameters) 
- [Snapshot code for a training run](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#snapshot-code-for-a-training-run) 
- [Write logs file to csv every k batches](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#write-logs-file-to-csv-every-k-batches)

###### Training loop    

- [Accumulate gradients](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#accumulated-gradients)
- [Force training for min or max epochs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-training-for-min-or-max-epochs)
- [Early stopping callback](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#early-stopping)    
- [Force disable early stop](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-disable-early-stop)
- [Gradient Clipping](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#gradient-clipping)
- [Hooks](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/)
- [Learning rate scheduling](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#configure_optimizers)
- [Use multiple optimizers (like GANs)](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#configure_optimizers)
- [Set how much of the training set to check (1-100%)](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#set-how-much-of-the-training-set-to-check)
- [Step optimizers at arbitrary intervals](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/#optimizer_step)

###### Validation loop    

- [Check validation every n epochs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#check-validation-every-n-epochs)
- [Hooks](https://williamfalcon.github.io/pytorch-lightning/Trainer/hooks/)
- [Set how much of the validation set to check](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](https://williamfalcon.github.io/pytorch-lightning/Trainer/Validation%20loop/#set-the-number-of-validation-sanity-steps)

###### Testing loop  
- [Run test set](https://williamfalcon.github.io/pytorch-lightning/Trainer/Testing%20loop/)  
