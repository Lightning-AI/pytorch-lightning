:orphan:

.. _logging_expert:

########################################
Track and Visualize Experiments (expert)
########################################
**Audience:** Users who want to make their own progress bars or integrate new experiment managers.

----

***********************
Change the progress bar
***********************

If you'd like to change the way the progress bar displays information you can use some of our built-in progress bard or build your own.

----

Use the TQDMProgressBar
=======================
To use the TQDMProgressBar pass it into the *callbacks* :class:`~lightning.pytorch.trainer.trainer.Trainer` argument.

.. code-block:: python

    from lightning.pytorch.callbacks import TQDMProgressBar

    trainer = Trainer(callbacks=[TQDMProgressBar()])

----

Use the RichProgressBar
=======================
The RichProgressBar can add custom colors and beautiful formatting for your progress bars. First, install the *`rich <https://github.com/Textualize/rich>`_*  library

.. code-block:: bash

    pip install rich

Then pass the callback into the callbacks :class:`~lightning.pytorch.trainer.trainer.Trainer` argument:

.. code-block:: python

    from lightning.pytorch.callbacks import RichProgressBar

    trainer = Trainer(callbacks=[RichProgressBar()])

The rich progress bar can also have custom themes

.. code-block:: python

    from lightning.pytorch.callbacks import RichProgressBar
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

    # create your own theme!
    theme = RichProgressBarTheme(description="green_yellow", progress_bar="green1")

    # init as normal
    progress_bar = RichProgressBar(theme=theme)
    trainer = Trainer(callbacks=progress_bar)

----

************************
Customize a progress bar
************************
To customize either the  :class:`~lightning.pytorch.callbacks.TQDMProgressBar` or the  :class:`~lightning.pytorch.callbacks.RichProgressBar`, subclass it and override any of its methods.

.. code-block:: python

    from lightning.pytorch.callbacks import TQDMProgressBar


    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description("running validation...")
            return bar

----

***************************
Build your own progress bar
***************************
To build your own progress bar, subclass :class:`~lightning.pytorch.callbacks.ProgressBar`

.. code-block:: python

    from lightning.pytorch.callbacks import ProgressBar


    class LitProgressBar(ProgressBar):
        def __init__(self):
            super().__init__()  # don't forget this :)
            self.enable = True

        def disable(self):
            self.enable = False

        def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)  # don't forget this :)
            percent = (self.train_batch_idx / self.total_train_batches) * 100
            sys.stdout.flush()
            sys.stdout.write(f"{percent:.01f} percent complete \r")


    bar = LitProgressBar()
    trainer = Trainer(callbacks=[bar])

----

*******************************
Integrate an experiment manager
*******************************
To create an integration between a custom logger and Lightning, subclass :class:`~lightning.pytorch.loggers.Logger`

.. code-block:: python

    from lightning.pytorch.loggers import Logger


    class LitLogger(Logger):
        @property
        def name(self) -> str:
            return "my-experiment"

        @property
        def version(self):
            return "version_0"

        def log_metrics(self, metrics, step=None):
            print("my logged metrics", metrics)

        def log_hyperparams(self, params, *args, **kwargs):
            print("my logged hyperparameters", params)
