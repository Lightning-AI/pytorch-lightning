.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _progress_bar:


Progress Bar
============

Lightning supports two different types of progress bars (`tqdm <https://github.com/tqdm/tqdm>`_ and `rich <https://github.com/Textualize/rich>`_). :class:`~pytorch_lightning.callbacks.TQDMProgressBar` is used by default,
but you can override it by passing a custom :class:`~pytorch_lightning.callbacks.TQDMProgressBar` or :class:`~pytorch_lightning.callbacks.RichProgressBar` to the ``callbacks`` argument of the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

You could also use the :class:`~pytorch_lightning.callbacks.ProgressBarBase` class to implement your own progress bar.

-------------

TQDMProgressBar
---------------

The :class:`~pytorch_lightning.callbacks.TQDMProgressBar` uses the `tqdm <https://github.com/tqdm/tqdm>`_ library internally and is the default progress bar used by Lightning.
It prints to ``stdout`` and shows up to four different bars:

- **sanity check progress:** the progress during the sanity check run
- **main progress:** shows training + validation progress combined. It also accounts for multiple validation runs during training when :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
- **validation progress:** only visible during validation; shows total progress over all validation datasets.
- **test progress:** only active when testing; shows total progress over all test datasets.

For infinite datasets, the progress bar never ends.

You can update ``refresh_rate`` (rate (number of batches) at which the progress bar get updated) for :class:`~pytorch_lightning.callbacks.TQDMProgressBar` by:

.. code-block:: python

    from pytorch_lightning.callbacks import TQDMProgressBar

    trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=10)])

If you want to customize the default :class:`~pytorch_lightning.callbacks.TQDMProgressBar` used by Lightning, you can override
specific methods of the callback class and pass your custom implementation to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

.. code-block:: python

    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description("running validation...")
            return bar


    trainer = Trainer(callbacks=[LitProgressBar()])

.. seealso::
    - :class:`~pytorch_lightning.callbacks.TQDMProgressBar` docs.
    - `tqdm library <https://github.com/tqdm/tqdm>`__

----------------

RichProgressBar
---------------

`Rich <https://github.com/Textualize/rich>`_ is a Python library for rich text and beautiful formatting in the terminal.
To use the :class:`~pytorch_lightning.callbacks.RichProgressBar` as your progress bar, first install the package:

.. code-block:: bash

    pip install rich

Then configure the callback and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. code-block:: python

    from pytorch_lightning.callbacks import RichProgressBar

    trainer = Trainer(callbacks=[RichProgressBar()])

Customize the theme for your :class:`~pytorch_lightning.callbacks.RichProgressBar` like this:

.. code-block:: python

    from pytorch_lightning.callbacks import RichProgressBar
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

    # create your own theme!
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    trainer = Trainer(callbacks=progress_bar)

You can customize the components used within :class:`~pytorch_lightning.callbacks.RichProgressBar` with ease by overriding the
:func:`~pytorch_lightning.callbacks.RichProgressBar.configure_columns` method.

.. code-block:: python

    from rich.progress import TextColumn

    custom_column = TextColumn("[progress.description]Custom Rich Progress Bar!")


    class CustomRichProgressBar(RichProgressBar):
        def configure_columns(self, trainer):
            return [custom_column]


    progress_bar = CustomRichProgressBar()

If you wish for a new progress bar to be displayed at the end of every epoch, you should enable
:paramref:`RichProgressBar.leave <pytorch_lightning.callbacks.RichProgressBar.leave>` by passing ``True``

.. code-block:: python

    from pytorch_lightning.callbacks import RichProgressBar

    trainer = Trainer(callbacks=[RichProgressBar(leave=True)])

.. seealso::
    - :class:`~pytorch_lightning.callbacks.RichProgressBar` docs.
    - :class:`~pytorch_lightning.callbacks.RichModelSummary` docs to customize the model summary table.
    - `Rich library <https://github.com/Textualize/rich>`__.


.. note::

    Progress bar is automatically enabled with the Trainer, and to disable it, one should do this:

    .. code-block:: python

        trainer = Trainer(enable_progress_bar=False)
