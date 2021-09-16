import logging
import signal

from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities.imports import _fault_tolerant_training

log = logging.getLogger(__name__)


class SignalConnector:
    def __init__(self, trainer):
        self.trainer = trainer
        self.trainer._should_gracefully_terminate = False

    def register_signal_handlers(self):
        cluster_env = getattr(self.trainer.training_type_plugin, "cluster_environment", None)
        if _fault_tolerant_training() and not isinstance(cluster_env, SLURMEnvironment):
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        self.trainer._should_gracefully_terminate = True

    def term_handler(self, signum, frame):  # pragma: no-cover
        log.info("bypassing sigterm")
