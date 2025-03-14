import logging
import os
import re
import signal
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Union

import lightning.pytorch as pl
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_info

# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]

log = logging.getLogger(__name__)


class _HandlersCompose:
    def __init__(self, signal_handlers: Union[list[_HANDLER], _HANDLER]) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)


class _SignalConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.received_sigterm = False
        self.trainer = trainer
        self._original_handlers: dict[_SIGNUM, _HANDLER] = {}

    def register_signal_handlers(self) -> None:
        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()

        sigusr_handlers: list[_HANDLER] = []
        sigterm_handlers: list[_HANDLER] = [self._sigterm_notifier_fn]

        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            log.info("SLURM auto-requeueing enabled. Setting signal handlers.")
            sigusr_handlers.append(self._slurm_sigusr_handler_fn)
            sigterm_handlers.append(self._sigterm_handler_fn)

        # Windows seems to have signal incompatibilities
        if not _IS_WINDOWS:
            sigusr = environment.requeue_signal if isinstance(environment, SLURMEnvironment) else signal.SIGUSR1
            assert sigusr is not None
            if sigusr_handlers and not self._has_already_handler(sigusr):
                self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))

            # we have our own handler, but include existing ones too
            if self._has_already_handler(signal.SIGTERM):
                sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
            self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(self.trainer.default_root_dir)
        self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer.is_global_zero:
            # find job id
            array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            if array_job_id is not None:
                array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
                job_id = f"{array_job_id}_{array_task_id}"
            else:
                job_id = os.environ["SLURM_JOB_ID"]

            assert re.match("[0-9_-]+", job_id)
            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            log.info(f"requeing job {job_id}...")
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                result = call(" ".join(cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f"Requeued SLURM job: {job_id}")
            else:
                log.warning(f"Requeuing SLURM job {job_id} failed with error code {result}")

    def _sigterm_notifier_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(rank_prefixed_message(f"Received SIGTERM: {signum}", self.trainer.local_rank))
        # subprocesses killing the parent process is not supported, only the parent (rank 0) does it
        if not self.received_sigterm:
            # send the same signal to the subprocesses
            launcher = self.trainer.strategy.launcher
            if launcher is not None:
                launcher.kill(signum)
        self.received_sigterm = True

    def _sigterm_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(f"Bypassing SIGTERM: {signum}")

    def teardown(self) -> None:
        """Restores the signals that were previously configured before :class:`_SignalConnector` replaced them."""
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                self._register_signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> dict[_SIGNUM, _HANDLER]:
        """Collects the currently assigned signal handlers."""
        valid_signals = _SignalConnector._valid_signals()
        if not _IS_WINDOWS:
            # SIGKILL and SIGSTOP are not allowed to be modified by the user
            valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> set[signal.Signals]:
        """Returns all valid signals supported on the current platform."""
        return signal.valid_signals()

    @staticmethod
    def _has_already_handler(signum: _SIGNUM) -> bool:
        return signal.getsignal(signum) not in (None, signal.SIG_DFL)

    @staticmethod
    def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)  # type: ignore[arg-type]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_original_handlers"] = {}
        return state


def _get_sigkill_signal() -> _SIGNUM:
    return signal.SIGTERM if _IS_WINDOWS else signal.SIGKILL
