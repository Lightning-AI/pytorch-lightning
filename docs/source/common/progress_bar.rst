Progress Bar
============

Lightning supports two different types of progress bars (`tqdm <https://github.com/tqdm/tqdm>`_ and `rich <https://github.com/Textualize/rich>`_). :class:`~pytorch_lightning.callbacks.TQDMProgressBar` is used by default,
but you can override it by passing custom :class:`~pytorch_lightning.callbacks.TQDMProgressBar` or :class:`~pytorch_lightning.callbacks.RichProgressBar` to the ``callbacks`` flag of :class:`~pytorch_lightning.trainer.trainer.Trainer`.

Supported Progress Bars
-----------------------

.. currentmodule:: pytorch_lightning.callbacks.progress

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    RichProgressBar
    TQDMProgressBar

-------

TQDMProgressBar
---------------

It uses the `tqdm <https://github.com/tqdm/tqdm>`_ library internally and is the default progress bar used by Lightning.
It prints to ``stdout`` and shows up to four different bars:

- **sanity check progress:** the progress during the sanity check run
- **main progress:** shows training + validation progress combined. It also accounts for multiple validation runs during training when :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
- **validation progress:** only visible during validation; shows total progress over all validation datasets.
- **test progress:** only active when testing; shows total progress over all test datasets.

For infinite datasets, the progress bar never ends.

You can update ``refresh_rate`` (rate (number of batches) at which the progress bar get updated) for :class:`~pytorch_lightning.callbacks.TQDMProgressBar` by:

.. code-block:: python

    from pytorch_lightning.callbacks import TQDMProgressBar

    trainer = Trainer(callbacks=TQDMProgressBar(refresh_rate=10))

If you want to customize the default :class:`~pytorch_lightning.callbacks.TQDMProgressBar` used by Lightning, you can override
specific methods of the callback class and pass your custom implementation to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

.. code-block:: python

    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description("running validation...")
            return bar


    bar = LitProgressBar()
    trainer = Trainer(callbacks=[bar])

.. seealso::
    - :class:`~pytorch_lightning.callbacks.TQDMProgressBar` docs.
    - `tqdm library <https://github.com/tqdm/tqdm>`__

----------------

RichProgressBar
---------------
