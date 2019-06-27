These flags are useful to help debug a model.

---
#### Fast dev run 
This flag is meant for debugging a full train/val/test loop. It'll activate callbacks, everything but only with 1 training and 1 validation batch.
Use this to debug a full run of your program quickly
``` {.python}
# DEFAULT
trainer = Trainer(fast_dev_run=False)
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
#### Print the parameter count by layer
By default lightning prints a list of parameters *and submodules* when it starts training.

---
#### Print which gradients are nan 
This option prints a list of tensors with nan gradients.
``` {.python}
# DEFAULT
trainer = Trainer(print_nan_grads=False)
```

---
#### Log GPU usage
Lightning automatically logs gpu usage to the test tube logs. It'll only do it at the metric logging interval, so it doesn't slow down training.