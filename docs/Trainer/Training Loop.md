The lightning training loop handles everything except the actual computations of your model. To decide what will happen in your training loop, define the [training_step function](../../Pytorch-lightning/LightningModule/#training_step).

Below are all the things lightning automates for you in the training loop.

---
#### Accumulated gradients  
Accumulated gradients runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN. 

``` {.python}
# DEFAULT (ie: no accumulated grads)
trainer = Trainer(accumulate_grad_batches=1)
```

---
#### Anneal Learning rate
Cut the learning rate by 10 at every epoch listed in this list.
``` {.python}
# DEFAULT (don't anneal)
trainer = Trainer(lr_scheduler_milestones=None)

# cut LR by 10 at 100, 200, and 300 epochs 
trainer = Trainer(lr_scheduler_milestones=[100, 200, 300])
```

---
#### Check GPU usage
Lightning automatically logs gpu usage to the test tube logs. It'll only do it at the metric logging interval, so it doesn't slow down training.

---
#### Check which gradients are nan 
This option prints a list of tensors with nan gradients.
``` {.python}
# DEFAULT
trainer = Trainer(print_nan_grads=False)
```

---
#### Display metrics in progress bar 
``` {.python}
# DEFAULT
trainer = Trainer(progress_bar=True)
```

---
#### Display the parameter count by layer
By default lightning prints a list of parameters *and submodules* when it starts training.

---
#### Fast dev run 
This flag is meant for debugging a full train/val/test loop. It'll activate callbacks, everything but only with 1 training and 1 validation batch.
Use this to debug a full run of your program quickly
``` {.python}
# DEFAULT
trainer = Trainer(fast_dev_run=False)
```

---
#### Force training for min or max epochs
It can be useful to force training for a minimum number of epochs or limit to a max number
``` {.python}
# DEFAULT
trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)
```

---
#### Force disable early stop 
Use this to turn off early stopping and run training to the [max_epoch](#force-training-for-min-or-max-epochs)
``` {.python}
# DEFAULT
trainer = Trainer(enable_early_stop=True)
```

---
#### Inspect gradient norms
Looking at grad norms can help you figure out where training might be going wrong.
``` {.python}
# DEFAULT (-1 doesn't track norms)
trainer = Trainer(track_grad_norm=-1)

# track the LP norm (P=2 here)
trainer = Trainer(track_grad_norm=2)
```


---
#### Make model overfit on subset of data
A useful debugging trick is to make your model overfit a tiny fraction of the data.
``` {.python}
# DEFAULT don't overfit (ie: normal training)
trainer = Trainer(overfit_pct=0.0)

# overfit on 1% of data 
trainer = Trainer(overfit_pct=0.01)
```

---
#### Process position
When running multiple models on the same machine we want to decide which progress bar to use.
Lightning will stack progress bars according to this value. 
``` {.python}
# DEFAULT
trainer = Trainer(process_position=0)

# if this is the second model on the node, show the second progress bar below
trainer = Trainer(process_position=1)
```


---
#### Set how much of the training set to check
If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag
``` {.python}
# DEFAULT
trainer = Trainer(train_percent_check=1.0)

# check 10% only
trainer = Trainer(train_percent_check=0.1)
```
