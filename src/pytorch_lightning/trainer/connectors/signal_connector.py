import logging
import os
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities.imports import _fault_tolerant_training, _IS_WINDOWS
from pytorch_lightning.utilities.rank_zero import rank_zero_info

# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]

log = logging.getLogger(__name__)


class HandlersCompose:
    def __init__(self, signal_handlers: Union[List[_HANDLER], _HANDLER]) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)


class SignalConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = trainer
        self.trainer._terminate_gracefully = False
        self._original_handlers: Dict[_SIGNUM, _HANDLER] = {}

    def register_signal_handlers(self) -> None:
        self._original_handlers = self._get_current_signal_handlers()

        sigusr1_handlers: List[_HANDLER] = []
        sigterm_handlers: List[_HANDLER] = []

        if _fault_tolerant_training():
            sigterm_handlers.append(self.fault_tolerant_sigterm_handler_fn)

        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            log.info("SLURM auto-requeueing enabled. Setting signal handlers.")
            sigusr1_handlers.append(self.slurm_sigusr1_handler_fn)
            sigterm_handlers.append(self.sigterm_handler_fn)

        # signal.SIGUSR1 doesn't seem available on windows
        if not self._is_on_windows():
            if sigusr1_handlers and not self._has_already_handler(signal.SIGUSR1):
                self._register_signal(signal.SIGUSR1, HandlersCompose(sigusr1_handlers))

            if sigterm_handlers and not self._has_already_handler(signal.SIGTERM):
                self._register_signal(signal.SIGTERM, HandlersCompose(sigterm_handlers))

    def slurm_sigusr1_handler_fn(self, signum: _SIGNUM, frame: FrameType) -> None:
        rank_zero_info("handling SIGUSR1")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")
        # TODO: in v1.8 change this to use self.trainer.default_root_dir
        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(self.trainer._weights_save_path_internal)
        self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer.is_global_zero:
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

    def fault_tolerant_sigterm_handler_fn(self, signum: _SIGNUM, frame: FrameType) -> None:
        log.info(f"Received signal {signum}. Saving a fault-tolerant checkpoint and terminating.")
        self.trainer._terminate_gracefully = True

    def sigterm_handler_fn(self, signum: _SIGNUM, frame: FrameType) -> None:
        log.info("bypassing sigterm")

    def teardown(self) -> None:
        """Restores the signals that were previously configured before :class:`SignalConnector` replaced them."""
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                self._register_signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> Dict[_SIGNUM, _HANDLER]:
        """Collects the currently assigned signal handlers."""
        valid_signals = SignalConnector._valid_signals()
        if not _IS_WINDOWS:
            # SIGKILL and SIGSTOP are not allowed to be modified by the user
            valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> Set[signal.Signals]:
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
                signal.SIGBREAK,
            }
        return set(signal.Signals)

    @staticmethod
    def _is_on_windows() -> bool:
        return sys.platform == "win32"

    @staticmethod
    def _has_already_handler(signum: _SIGNUM) -> bool:
        return signal.getsignal(signum) not in (None, signal.SIG_DFL)

    @staticmethod
    def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)  # type: ignore[arg-type]

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_original_handlers"] = {}
        return state
