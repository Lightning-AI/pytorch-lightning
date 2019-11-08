Lighting offers options for logging information about model, gpu usage, etc, via several different logging frameworks. It also offers printing options for training monitoring.

---   
### default_save_path   
Lightning sets a default TestTubeLogger and CheckpointCallback for you which log to
```os.getcwd()``` by default. To modify the logging path you can set:
```python
Trainer(default_save_path='/your/path/to/save/checkpoints')
```
 
If you need more custom behavior (different paths for both, different metrics, etc...)
from the logger and the checkpointCallback, pass in your own instances as explained below.


---
### Setting up logging

The trainer inits a default logger for you (TestTubeLogger). All logs will
go to the current working directory under a folder named ```os.getcwd()/lightning_logs``. 

If you want to modify the default logging behavior even more, pass in a logger
(which should inherit from `LightningBaseLogger`).   

```{.python}
my_logger = MyLightningLogger(...)
trainer = Trainer(logger=my_logger)
```

The path in this logger will overwrite default_save_path.

Lightning supports several common experiment tracking frameworks out of the box

---
#### Test tube

Log using [test tube](https://williamfalcon.github.io/test-tube/). Test tube logger is
a strict subclass of [PyTorch SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html), refer to their
documentation for all supported operations. The TestTubeLogger adds a nicer folder structure
to manage experiments and snapshots all hyperparameters you pass to a LightningModule.

```{.python}
from pytorch_lightning.logging import TestTubeLogger
tt_logger = TestTubeLogger(
    save_dir=".",
    name="default",
    debug=False,
    create_git_tag=False
)
trainer = Trainer(logger=tt_logger)
```

Use the logger anywhere in you LightningModule as follows:
```python
def train_step(...):
    # example
    self.logger.experiment.whatever_method_summary_writer_supports(...)
    
def any_lightning_module_function_or_hook(...):
    self.logger.experiment.add_histogram(...)
```

---
#### MLFlow

Log using [mlflow](https://mlflow.org)

```{.python}
from pytorch_lightning.logging import MLFlowLogger
mlf_logger = MLFlowLogger(
    experiment_name="default",
    tracking_uri="file:/."
)
trainer = Trainer(logger=mlf_logger)
```
Use the logger anywhere in you LightningModule as follows:
```python
def train_step(...):
    # example
    self.logger.experiment.whatever_ml_flow_supports(...)
    
def any_lightning_module_function_or_hook(...):
    self.logger.experiment.whatever_ml_flow_supports(...)
```

---
#### Comet.ml

Log using [comet](https://www.comet.ml)

```{.python}
from pytorch_lightning.logging import CometLogger
# arguments made to CometLogger are passed on to the comet_ml.Experiment class
comet_logger = CometLogger(
    api_key=os.environ["COMET_KEY"],
    workspace=os.environ["COMET_WORKSPACE"],
    project_name="default_project", # Optional
    rest_api_key=os.environ["COMET_REST_KEY"], # Optional
    experiment_name="default" # Optional
)
trainer = Trainer(logger=comet_logger)
```
Use the logger anywhere in you LightningModule as follows:
```python
def train_step(...):
    # example
    self.logger.experiment.whatever_comet_ml_supports(...)

def any_lightning_module_function_or_hook(...):
    self.logger.experiment.whatever_comet_ml_supports(...)
```

---
#### Custom logger

You can implement your own logger by writing a class that inherits from
`LightningLoggerBase`. Use the `rank_zero_only` decorator to make sure that
only the first process in DDP training logs data.

```{.python}
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

class MyLogger(LightningLoggerBase):

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass
    
    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass
    
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass
    
    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
```

If you write a logger than may be useful to others, please send
a pull request to add it to Lighting!

---
#### Using loggers
You can call the logger anywhere from your LightningModule by doing:
```python
def train_step(...):
    # example
    self.logger.experiment.whatever_method_summary_writer_supports(...)
    
def any_lightning_module_function_or_hook(...):
    self.logger.experiment.add_histogram(...)
```

#### Display metrics in progress bar 
``` {.python}
# DEFAULT
trainer = Trainer(show_progress_bar=True)
```

---
#### Log metric row every k batches 
Every k batches lightning will make an entry in the metrics log
``` {.python}
# DEFAULT (ie: save a .csv log file every 10 batches)
trainer = Trainer(row_log_interval=10)
```   

---
#### Log GPU memory
Logs GPU memory when metrics are logged.   
``` {.python}
# DEFAULT
trainer = Trainer(log_gpu_memory=None)

# log only the min/max utilization
trainer = Trainer(log_gpu_memory='min_max')

# log all the GPU memory (if on DDP, logs only that node)
trainer = Trainer(log_gpu_memory='all')
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
Automatically log hyperparameters stored in the `hparams` attribute as an `argparse.Namespace` 
``` {.python}

class MyModel(pl.Lightning):
    def __init__(self, hparams):
        self.hparams = hparams

    ...

args = parser.parse_args()
model = MyModel(args)

logger = TestTubeLogger(...)
t = Trainer(logger=logger)
trainer.fit(model)
```

---
#### Write logs file to csv every k batches 
Every k batches, lightning will write the new logs to disk
``` {.python}
# DEFAULT (ie: save a .csv log file every 100 batches)
trainer = Trainer(log_save_interval=100)
```

