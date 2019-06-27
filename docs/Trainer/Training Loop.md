The asdf

---
#### Accumulated gradients  
Accumulated gradients runs K small batches of size N before doing a backwards pass. The effect is a large effective batch size of size KxN. 

``` {.python}
# default 1 (ie: no accumulated grads)
trainer = Trainer(accumulate_grad_batches=1)
```

---
#### Check GPU usage
Lightning automatically logs gpu usage to the test tube logs. It'll only do it at the metric logging interval, so it doesn't slow down training.

---
#### Check which gradients are nan 
This option prints a list of tensors with nan gradients.
``` {.python}
trainer = Trainer(print_nan_grads=False)
```

---
#### Check validation every n epochs 
If you have a small dataset you might want to check validation every n epochs
``` {.python}
trainer = Trainer(check_val_every_n_epoch=1)
```

---
#### Display metrics in progress bar 
``` {.python}
trainer = Trainer(progress_bar=True)
```

---
#### Display the parameter count by layer
By default lightning prints a list of parameters *and submodules* when it starts training.

---
#### Force training for min or max epochs
It can be useful to force training for a minimum number of epochs or limit to a max number
``` {.python}
trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1000)
```
