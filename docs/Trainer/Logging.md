Lighting offers a few options for logging information about model, gpu usage, etc (via test-tube). It also offers printing options for training monitoring.


---
#### Display metrics in progress bar 
``` {.python}
# DEFAULT
trainer = Trainer(progress_bar=True)
```

---
#### Log metric row every k batches 
Every k batches lightning will make an entry in the metrics log
``` {.python}
# DEFAULT (ie: save a .csv log file every 10 batches)
trainer = Trainer(add_log_row_interval=10)
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
#### Save a snapshot of all hyperparameters 
Whenever you call .save() on the test-tube experiment it logs all the hyperparameters in current use.
Give lightning a test-tube Experiment object to automate this for you.
``` {.python}
from test-tube import Experiment

exp = Experiment(...)
Trainer(experiment=exp)
```

---
#### Snapshot code for a training run
Whenever you call .save() on the test-tube experiment it snapshows all code and pushes to a git tag.
Give lightning a test-tube Experiment object to automate this for you.
``` {.python}
from test-tube import Experiment

exp = Experiment(create_git_tag=True)
Trainer(experiment=exp)
```


---
#### Write logs file to csv every k batches 
Every k batches, lightning will write the new logs to disk
``` {.python}
# DEFAULT (ie: save a .csv log file every 100 batches)
trainer = Trainer(log_save_interval=100)
```

