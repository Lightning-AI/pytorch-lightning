import logging
import os
import signal
from subprocess import call
from typing import Callable, List, Optional, Union

from pytorch_lightning.utilities.imports import _fault_tolerant_training

log = logging.getLogger(__name__)


class HandlersCompose:
    def __init__(self, signal_handlers: Union[List[Callable], Callable]):
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum, frame):
        for signal_handler in self.signal_handlers:
            signal_handler(signum, frame)


class SignalConnector:
    def __init__(self, trainer, sigusr1_handler: Optional[Callable] = None, sigterm_handler: Optional[Callable] = None):
        self.trainer = trainer
        self.trainer._terminate_gracefully = False
        self._sigusr1_handler = sigusr1_handler
        self._sigterm_handler = sigterm_handler

    @property
    def sigusr1_handler(self) -> Optional[Callable]:
        return self._sigusr1_handler

    @sigusr1_handler.setter
    def sigusr1_handler(self, sigusr1_handler: Callable) -> None:
        self._sigusr1_handler = sigusr1_handler

    @property
    def sigterm_handler(self) -> Optional[Callable]:
        return self._sigterm_handler

    @sigterm_handler.setter
    def sigterm_handler(self, sigterm_handler: Callable) -> None:
        self._sigterm_handler = sigterm_handler

    def register_signal_handlers(self):
        sigusr1_handlers = []
        sigterm_handlers = []

        if _fault_tolerant_training():
            sigusr1_handlers.append(self.fault_tolerant_sigusr1_handler_fn)

        if self._is_on_slurm():
            log.info("Set SLURM handle signals.")
            sigusr1_handlers.append(self.slurm_sigusr1_handler_fn)

        sigterm_handlers.append(self.sigterm_handler_fn)

        signal.signal(signal.SIGUSR1, HandlersCompose(self.sigusr1_handler or sigusr1_handlers))
        signal.signal(signal.SIGTERM, HandlersCompose(self.sigterm_handler or sigterm_handlers))

    def _is_on_slurm(self) -> bool:
        # see if we're using slurm (not interactive)
        on_slurm = False
        try:
            job_name = os.environ["SLURM_JOB_NAME"]
            if job_name != "bash":
                on_slurm = True
        # todo: specify the possible exception
        except Exception:
            pass

        return on_slurm

    def slurm_sigusr1_handler_fn(self, signum, frame):  # pragma: no-cover
        if self.trainer.is_global_zero:
            # save weights
            log.info("handling SIGUSR1")
            self.trainer.checkpoint_connector.hpc_save(self.trainer.weights_save_path, self.trainer.logger)

            # find job id
            job_id = os.environ["SLURM_JOB_ID"]
            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            log.info(f"requeing job {job_id}...")
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                joint_cmd = [str(x) for x in cmd]
                result = call(" ".join(joint_cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f"requeued exp {job_id}")
            else:
                log.warning("requeue failed...")

            # close experiment to avoid issues
            self.trainer.logger.close()

    def fault_tolerant_sigusr1_handler_fn(self, signum, frame):  # pragma: no-cover
        self.trainer._terminate_gracefully = True

    def sigterm_handler_fn(self, signum, frame):  # pragma: no-cover
        log.info("bypassing sigterm")
