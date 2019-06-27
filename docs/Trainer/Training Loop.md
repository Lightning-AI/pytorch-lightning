The asdf

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
#### Check validation every n epochs 
If you have a small dataset you might want to check validation every n epochs
``` {.python}
# DEFAULT
trainer = Trainer(check_val_every_n_epoch=1)
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
#### Force training for min or max epochs
It can be useful to force training for a minimum number of epochs or limit to a max number
``` {.python}
# DEFAULT
trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)
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
#### Set how much of the training set to check
If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag
``` {.python}
# DEFAULT
trainer = Trainer(train_percent_check=1.0)

# check 10% only
trainer = Trainer(train_percent_check=0.1)
```
