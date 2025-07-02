.. testsetup:: *

    from lightning.pytorch.trainer.trainer import Trainer

.. _progress_bar:


Customize the progress bar
==========================

Lightning supports two different types of progress bars (`tqdm <https://github.com/tqdm/tqdm>`_ and `rich <https://github.com/Textualize/rich>`_). :class:`~lightning.pytorch.callbacks.TQDMProgressBar` is used by default,
but you can override it by passing a custom :class:`~lightning.pytorch.callbacks.TQDMProgressBar` or :class:`~lightning.pytorch.callbacks.RichProgressBar` to the ``callbacks`` argument of the :class:`~lightning.pytorch.trainer.trainer.Trainer`.

You could also use the :class:`~lightning.pytorch.callbacks.ProgressBar` class to implement your own progress bar.

-------------

TQDMProgressBar
---------------

The :class:`~lightning.pytorch.callbacks.TQDMProgressBar` uses the `tqdm <https://github.com/tqdm/tqdm>`_ library internally and is the default progress bar used by Lightning.
It prints to ``stdout`` and shows up to four different bars:

- **sanity check progress:** the progress during the sanity check run
- **train progress:** shows the training progress. It will pause if validation starts and will resume when it ends, and also accounts for multiple validation runs during training when :paramref:`~lightning.pytorch.trainer.trainer.Trainer.val_check_interval` is used.
- **validation progress:** only visible during validation; shows total progress over all validation datasets.
- **test progress:** only active when testing; shows total progress over all test datasets.

For infinite datasets, the progress bar never ends.

You can update ``refresh_rate`` (rate (number of batches) at which the progress bar get updated) for :class:`~lightning.pytorch.callbacks.TQDMProgressBar` by:

.. code-block:: python

    from lightning.pytorch.callbacks import TQDMProgressBar

    trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=10)])

.. note::

    The ``smoothing`` option has no effect when using the default implementation of :class:`~lightning.pytorch.callbacks.TQDMProgressBar`, as the progress bar is updated using the ``bar.refresh()`` method instead of ``bar.update()``. This can cause the progress bar to become desynchronized with the actual progress. To avoid this issue, you can use the ``bar.update()`` method instead, but this may require customizing the :class:`~lightning.pytorch.callbacks.TQDMProgressBar` class.

By default the training progress bar is reset (overwritten) at each new epoch.
If you wish for a new progress bar to be displayed at the end of every epoch, set
:paramref:`TQDMProgressBar.leave <lightning.pytorch.callbacks.TQDMProgressBar.leave>` to ``True``.

.. code-block:: python

    trainer = Trainer(callbacks=[TQDMProgressBar(leave=True)])

If you want to customize the default :class:`~lightning.pytorch.callbacks.TQDMProgressBar` used by Lightning, you can override
specific methods of the callback class and pass your custom implementation to the :class:`~lightning.pytorch.trainer.trainer.Trainer`.

.. code-block:: python

    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description("running validation...")
            return bar


    trainer = Trainer(callbacks=[LitProgressBar()])

.. seealso::
    - :class:`~lightning.pytorch.callbacks.TQDMProgressBar` docs.
    - `tqdm library <https://github.com/tqdm/tqdm>`__

----------------

RichProgressBar
---------------

`Rich <https://github.com/Textualize/rich>`_ is a Python library for rich text and beautiful formatting in the terminal.
To use the :class:`~lightning.pytorch.callbacks.RichProgressBar` as your progress bar, first install the package:

.. code-block:: bash

    pip install rich

Then configure the callback and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. code-block:: python

    from lightning.pytorch.callbacks import RichProgressBar

    trainer = Trainer(callbacks=[RichProgressBar()])

Customize the theme for your :class:`~lightning.pytorch.callbacks.RichProgressBar` like this:

.. code-block:: python

    from lightning.pytorch.callbacks import RichProgressBar
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

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
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )

    trainer = Trainer(callbacks=progress_bar)

You can customize the components used within :class:`~lightning.pytorch.callbacks.RichProgressBar` with ease by overriding the
:func:`~lightning.pytorch.callbacks.RichProgressBar.configure_columns` method.

.. code-block:: python

    from rich.progress import TextColumn

    custom_column = TextColumn("[progress.description]Custom Rich Progress Bar!")


    class CustomRichProgressBar(RichProgressBar):
        def configure_columns(self, trainer):
            return [custom_column]


    progress_bar = CustomRichProgressBar()

If you wish for a new progress bar to be displayed at the end of every epoch, you should enable
:paramref:`RichProgressBar.leave <lightning.pytorch.callbacks.RichProgressBar.leave>` by passing ``True``

.. code-block:: python

    from lightning.pytorch.callbacks import RichProgressBar

    trainer = Trainer(callbacks=[RichProgressBar(leave=True)])

.. seealso::
    - :class:`~lightning.pytorch.callbacks.RichProgressBar` docs.
    - :class:`~lightning.pytorch.callbacks.RichModelSummary` docs to customize the model summary table.
    - `Rich library <https://github.com/Textualize/rich>`__.


.. note::

    Progress bar is automatically enabled with the Trainer, and to disable it, one should do this:

    .. code-block:: python

        trainer = Trainer(enable_progress_bar=False)
