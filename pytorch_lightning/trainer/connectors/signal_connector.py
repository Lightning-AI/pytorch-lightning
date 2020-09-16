import signal

from pytorch_lightning.trainer.states import TrainerState


class SignalConnector:
    """
    Takes care of registering and restoring signal handlers for the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`. By default, it handles
    SIGTERM, SIGINT and SIGSEGV by raising KeyboardInterrupt and letting Trainer do graceful shutdown.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.original_handlers = {}

    def setup(self):
        """ Registers the default signal handlers for the Trainer. """
        self.register_signal(signal.SIGTERM, self.default_teardown)
        self.register_signal(signal.SIGSEGV, self.default_teardown)
        self.register_signal(signal.SIGINT, self.default_teardown)

    def restore_signals(self):
        """ Restores the original signal handlers (e.g. the Python or user defaults) """
        for signum, handler in self.original_handlers.items():
            signal.signal(signum, handler)

    def register_signal(self, signum, handler):
        """ Registers a signal handler and saves a reference to the original handler. """
        self.original_handlers.update({signum: signal.getsignal(signum)})
        signal.signal(signum, handler)

    def default_teardown(self, signum, frame):  # pragma: no-cover
        """ This default teardown raises KeyboardInterrupt and lets Trainer handle the graceful shutdown. """
        if not self.trainer.interrupted:
            self.trainer.interrupted = True
            self.trainer._state = TrainerState.INTERRUPTED
            raise KeyboardInterrupt
