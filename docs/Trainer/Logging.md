


---
#### Display metrics in progress bar 
``` {.python}
# DEFAULT
trainer = Trainer(progress_bar=True)
```



---
#### Print which gradients are nan 
This option prints a list of tensors with nan gradients.
``` {.python}
# DEFAULT
trainer = Trainer(print_nan_grads=False)
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
