import logging
import os
import signal
import sys
import threading
from signal import Signals
from subprocess import call
from types import FrameType, FunctionType
from typing import Any, Callable, Dict, List, Set, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.imports import _fault_tolerant_training, _IS_WINDOWS

log = logging.getLogger(__name__)

_SIGNAL_HANDLER_DICT = Dict[Signals, Union[Callable[[Signals, FrameType], Any], int, None]]


class HandlersCompose:
    def __init__(self, signal_handlers: Union[List[Callable], Callable]) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: Signals, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            signal_handler(signum, frame)


class SignalConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = trainer
        self.trainer._terminate_gracefully = False
        self._original_handlers: _SIGNAL_HANDLER_DICT = {}

    def register_signal_handlers(self) -> None:
        self._original_handlers = self._get_current_signal_handlers()

        sigusr1_handlers: List[Callable] = []
        sigterm_handlers: List[Callable] = []

        if _fault_tolerant_training():
            sigusr1_handlers.append(self.fault_tolerant_sigusr1_handler_fn)

        if self._is_on_slurm():
            log.info("Set SLURM handle signals.")
            sigusr1_handlers.append(self.slurm_sigusr1_handler_fn)
            sigterm_handlers.append(self.sigterm_handler_fn)

        # signal.SIGUSR1 doesn't seem available on windows
        if not _IS_WINDOWS:
            if sigusr1_handlers and not self._has_already_handler(signal.SIGUSR1):
                self._register_signal(signal.SIGUSR1, HandlersCompose(sigusr1_handlers))

            if sigterm_handlers and not self._has_already_handler(signal.SIGTERM):
                self._register_signal(signal.SIGTERM, HandlersCompose(sigterm_handlers))

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
            if self.trainer.logger:
                self.trainer.logger.finalize("finished")

    def fault_tolerant_sigusr1_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        self.trainer._terminate_gracefully = True

    def sigterm_handler_fn(self, signum: Signals, frame: FrameType) -> None:
        log.info("bypassing sigterm")

    def teardown(self) -> None:
        """Restores the signals that were previsouly configured before :class:`SignalConnector` replaced them."""
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                signal.signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> _SIGNAL_HANDLER_DICT:
        """Collects the currently assigned signal handlers."""
        valid_signals = SignalConnector._valid_signals()
        if not _IS_WINDOWS:
            # SIGKILL and SIGSTOP are not allowed to be modified by the user
            valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> Set[Signals]:
        """Returns all valid signals supported on the current platform.

        Behaves identically to :func:`signals.valid_signals` in Python 3.8+ and implements the equivalent behavior for
        older Python versions.
        """
        if sys.version_info >= (3, 8):
            return signal.valid_signals()
        elif _IS_WINDOWS:
            # supported signals on Windows: https://docs.python.org/3/library/signal.html#signal.signal
            return {
                signal.SIGABRT,
                signal.SIGFPE,
                signal.SIGILL,
                signal.SIGINT,
                signal.SIGSEGV,
                signal.SIGTERM,
            }
        return set(signal.Signals)

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

    @staticmethod
    def _has_already_handler(signum: Signals) -> bool:
        try:
            return isinstance(signal.getsignal(signum), FunctionType)
        except AttributeError:
            return False

    @staticmethod
    def _register_signal(signum: Signals, handlers: HandlersCompose) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_original_handlers"] = {}
        return state
