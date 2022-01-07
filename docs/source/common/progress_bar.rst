Progress Bar
============

Lightning supports two different types of progress bars (tqdm and rich). :class:`~pytorch_lightning.callbacks.TQDMProgressBar` is used by default,
but you can override it by passing custom TQDMProgressBar or RichProgressBar to the ``callbacks`` flag of Trainer.
