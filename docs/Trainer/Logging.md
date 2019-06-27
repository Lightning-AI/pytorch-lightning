


---
#### Display metrics in progress bar 
``` {.python}
# DEFAULT
trainer = Trainer(progress_bar=True)
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
#### Print which gradients are nan 
This option prints a list of tensors with nan gradients.
``` {.python}
# DEFAULT
trainer = Trainer(print_nan_grads=False)
```

---
#### Save a snapshot of all hyperparameters 
Whenever you call .save() on the test-tube experiment it logs all the hyperparameters in current use.
Give lightning a test-tube Experiment object to automate this for you.

---
#### Log metric row every k batches 
Every k batches lightning will make an entry in the metrics log
``` {.python}
# DEFAULT (ie: save a .csv log file every 100 batches)
trainer = Trainer(add_log_row_interval=10)
```

---
#### Write logs file to csv every k batches 
Every k batches, lightning will write the new logs to disk
``` {.python}
# DEFAULT (ie: save a .csv log file every 100 batches)
trainer = Trainer(log_save_interval=100)
```

