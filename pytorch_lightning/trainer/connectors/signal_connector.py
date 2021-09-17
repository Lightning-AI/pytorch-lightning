import logging
import os
import signal
import sys
from signal import Signals
from subprocess import call
from types import FrameType, FunctionType
from typing import Callable, List, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.imports import _fault_tolerant_training

log = logging.getLogger(__name__)


class HandlersCompose:
    def __init__(self, signal_handlers: Union[List[Callable], Callable]):
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: Signals, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            signal_handler(signum, frame)


class SignalConnector:
    def __init__(self, trainer: "pl.Trainer"):
        self.trainer = trainer
        self.trainer._terminate_gracefully = False

    def register_signal_handlers(self) -> None:
        sigusr1_handlers: List[Callable] = []
        sigterm_handlers: List[Callable] = []

        if _fault_tolerant_training():
            sigusr1_handlers.append(self.fault_tolerant_sigusr1_handler_fn)

        if self._is_on_slurm():
            log.info("Set SLURM handle signals.")
            sigusr1_handlers.append(self.slurm_sigusr1_handler_fn)

        sigterm_handlers.append(self.sigterm_handler_fn)

        # signal.SIGUSR1 doesn't seem available on windows
        if not self._is_on_windows():
            if not self._has_already_handler(signal.SIGUSR1):
                signal.signal(signal.SIGUSR1, HandlersCompose(sigusr1_handlers))

            if not self._has_already_handler(signal.SIGTERM):
                signal.signal(signal.SIGTERM, HandlersCompose(sigterm_handlers))

    def slurm_sigusr1_handler_fn(self, signum: Signals, frame: FrameType) -> None:
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

    def fault_tolerant_sigusr1_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        self.trainer._terminate_gracefully = True

    def sigterm_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        log.info("bypassing sigterm")

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

    def _is_on_windows(self) -> bool:
        return sys.platform == "win32"

    def _has_already_handler(self, signum: Signals) -> bool:
        try:
            return isinstance(signal.getsignal(signum), FunctionType)
        except AttributeError:
            return False
