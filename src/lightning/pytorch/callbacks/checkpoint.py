from lightning.pytorch.callbacks.callback import Callback


class Checkpoint(Callback):
    r"""
    This is the base class for model checkpointing. Expert users may want to subclass it in case of writing
    custom :class:`~lightning.pytorch.callbacksCheckpoint` callback, so that
    the trainer recognizes the custom class as a checkpointing callback.
    """
