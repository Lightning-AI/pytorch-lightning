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

**Training loop**    

- [Accumulate gradients](Training%20Loop/#accumulated-gradients)
- [Anneal Learning rate](Training%20Loop/#anneal-learning-rate)
- [Check GPU usage](Training%20Loop/#Check-gpu-usage)
- [Check which gradients are nan](Training%20Loop/#check-which-gradients-are-nan)
- [Display metrics in progress bar](Training%20Loop/#display-metrics-in-progress-bar)
- [Display the parameter count by layer](Training%20Loop/#display-the-parameter-count-by-layer)
- [Fast dev run](Training%20Loop/#fast-dev-run)
- [Force training for min or max epochs](Training%20Loop/#force-training-for-min-or-max-epochs)
- [Force disable early stop](Training%20Loop/#force-disable-early-stop)
- [Inspect gradient norms](Training%20Loop/#inspect-gradient-norms)
- [Make model overfit on subset of data](Training%20Loop/#make-model-overfit-on-subset-of-data)
- [Use multiple optimizers (like GANs)](../Pytorch-lightning/LightningModule/#configure_optimizers)
- [Process position](Training%20Loop/#process-position)
- [Set how much of the training set to check (1-100%)](Training%20Loop/#set-how-much-of-the-training-set-to-check)

**Validation loop**    

- [Check validation every n epochs](Validation%20Loop/#check-validation-every-n-epochs)
- [Set how much of the validation set to check](Validation%20Loop/#set-how-much-of-the-validation-set-to-check)
- [Set how much of the test set to check](Validation%20Loop/#set-how-much-of-the-test-set-to-check)
- [Set validation check frequency within 1 training epoch](Validation%20Loop/#set-validation-check-frequency-within-1-training-epoch)
- [Set the number of validation sanity steps](Validation%20Loop/#set-the-number-of-validation-sanity-steps)
- [Check validation every n epochs](Validation%20Loop/#check-validation-every-n-epochs)

**Distributed training**    

- Single-gpu      
- Multi-gpu      
- Multi-node   
- 16-bit mixed precision

**Checkpointing**    

- Model saving
- Model loading 

**Computing cluster (SLURM)**    

- Automatic checkpointing   
- Automatic saving, loading  
- Running grid search on a cluster 
- Walltime auto-resubmit   