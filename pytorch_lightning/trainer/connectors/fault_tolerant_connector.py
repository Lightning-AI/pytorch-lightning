import logging
import signal

from pytorch_lightning.utilities.imports import _fault_tolerant_training

log = logging.getLogger(__name__)


class FaultTolerantConnector:
    def __init__(self, trainer):
        self.trainer = trainer
        self.trainer._should_gracefully_terminate = False

    def register_fault_tolerant_signal_handlers(self):
        if _fault_tolerant_training():
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        self.trainer._should_gracefully_terminate = True

    def term_handler(self, signum, frame):  # pragma: no-cover
        log.info("bypassing sigterm")
