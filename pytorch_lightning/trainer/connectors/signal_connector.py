import signal

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.trainer.states import TrainerState


class SignalConnector:
    """
    Takes care of registering and restoring signal handlers for the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`. This includes handling
    graceful shutdown for KeyboardInterrupt or SLURM autoresubmit signals.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.original_handlers = {}

    def setup(self):
        """
        Registers the signal handlers for the Trainer, including
        - training teardown signal handlers that run on interpreter exit and other POSIX signals.
        - HPC signal handling, i.e., registering auto-resubmit when on SLURM
        """
        self.register_signal(signal.SIGTERM, self.default_teardown)
        self.register_signal(signal.SIGSEGV, self.default_teardown)
        self.register_signal(signal.SIGINT, self.default_teardown)
        # atexit.register(self.trainer.run_training_teardown)

    def restore_signals(self):
        """ Restores the original signal handlers (e.g. the Python defaults) """
        for signum, handler in self.original_handlers.items():
            signal.signal(signum, handler)

    def register_signal(self, signum, handler):
        """ Registers a signal handler and saves a reference to the original handler. """
        self.original_handlers.update({signum: signal.getsignal(signum)})
        signal.signal(signum, handler)

    def default_teardown(self, signum, frame):  # pragma: no-cover
        """ Handles teardown for certain signals that interrupt training. """
        # trainer = self.trainer
        if not self.trainer.interrupted:
            self.trainer.interrupted = True
            self.trainer._state = TrainerState.INTERRUPTED
            # trainer.on_keyboard_interrupt()
            raise KeyboardInterrupt
