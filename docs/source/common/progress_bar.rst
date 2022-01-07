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

TQDMProgressBar
---------------