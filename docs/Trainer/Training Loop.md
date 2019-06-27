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
trainer = Trainer(check_grad_nans=False)
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