# Trainer
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/models/trainer.py)]

The lightning trainer abstracts best practices for running a training, val, test routine. It calls parts of your model when it wants to hand over full control and otherwise makes training assumptions which are now standard practice in AI research.

This is the basic use of the trainer:

``` {.python}
from pytorch_lightning import Trainer

model = LightningTemplate()

trainer = Trainer()
trainer.fit(model)
```

But of course the fun is in all the advanced things it can do:


**Checkpointing**    

- Model saving
- Model loading 

**Computing cluster (SLURM)**    

- [Running grid search on a cluster](SLURM%20Managed%20Cluster/#running-grid-search-on-a-cluster)
- [Walltime auto-resubmit](SLURM%20Managed%20Cluster/#walltime-auto-resubmit)   

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
- [Log metric row every k batches](Logging/#log-metric-row-every-k-batches)
- [Process position](Logging/#process-position)
- [Tensorboard support](https://williamfalcon.github.io/pytorch-lightning/Trainer/Logging/#tensorboard-support)
- [Save a snapshot of all hyperparameters](Logging/#save-a-snapshot-of-all-hyperparameters) 
- [Snapshot code for a training run](Logging/#snapshot-code-for-a-training-run) 
- [Write logs file to csv every k batches](Logging/#write-logs-file-to-csv-every-k-batches)

**Training loop**    

- [Accumulate gradients](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#accumulated-gradients)
- [Force training for min or max epochs](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-training-for-min-or-max-epochs)
- [Force disable early stop](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#force-disable-early-stop)
- [Gradient Clipping](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#gradient-clipping)
- [Hooks](hooks)
- [Learning rate scheduling](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#configure_optimizers)
- [Use multiple optimizers (like GANs)](https://williamfalcon.github.io/pytorch-lightning/Pytorch-Lightning/LightningModule/#configure_optimizers)
- [Set how much of the training set to check (1-100%)](https://williamfalcon.github.io/pytorch-lightning/Trainer/Training%20Loop/#set-how-much-of-the-training-set-to-check)

**Validation loop**    

- [Check validation every n epochs](Validation%20Loop/#check-validation-every-n-epochs)
- [Hooks](hooks)
- [Set how much of the validation set to check](Validation%20Loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](Validation%20Loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](Validation%20Loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](Validation%20Loop/#set-the-number-of-validation-sanity-steps)
