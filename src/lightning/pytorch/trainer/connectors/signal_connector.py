import logging
import os
import signal
import subprocess
import threading
from types import FrameType
from typing import Any, Callable, Union

import lightning.pytorch as pl
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_debug, rank_zero_info, rank_zero_warn

# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType | None], Any], int, signal.Handlers, None]

log = logging.getLogger(__name__)


class _HandlersCompose:
    def __init__(self, signal_handlers: list[_HANDLER]) -> None:
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType | None) -> None:
        for signal_handler in self.signal_handlers:
            if signal_handler is signal.SIG_DFL or signal_handler is signal.SIG_IGN:
                # If the handler is ignore, we skip it. Since there is no way for us to
                # trigger, the default signal handler, we ignore that one too
                continue
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)


class _SignalFlag:
    """Becomes true when called as a signal handler."""

    def __init__(self) -> None:
        self.state = False

    def __call__(self, signum: _SIGNUM, _: FrameType | None) -> None:
        self.set()

    def set(self) -> None:
        self.state = True

    def check_and_reset(self) -> bool:
        """Check the flag and reset it to false."""
        state = self.state
        self.state = False
        return state


class _SignalHandlerCallback(Callback):
    def __init__(self, connector: "_SignalConnector") -> None:
        self.connector = connector

        # Register same method for all callback methods
        on_methods = [f for f in dir(Callback) if f.startswith("on_") and callable(getattr(Callback, f))]
        for f in ["setup", "teardown"] + on_methods:
            setattr(self, f, self.notify_connector)

    def notify_connector(self, *args: Any, **kwargs: Any) -> None:
        self.connector._process_signals()


class _SignalConnector:
    """Listen for process signals to, for example, requeue a job on a SLURM cluster.

    The connector only stores the reception of signals in flags and then processes them at the next possible opportunity
    in the current loop. This minimizes the amount of code running in signal handlers, because file IO in signal
    handlers can crash the process. This also guarantees that we are not checkpointing in the middle of a backward pass
    or similar.

    """

    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = trainer

        # This flag is checked by the trainer and loops to exit gracefully
        self.received_sigterm = False

        self.sigterm_flag = _SignalFlag()
        self.requeue_flag = _SignalFlag()

        self._original_handlers: dict[_SIGNUM, _HANDLER] = {}

    def register_callback(self) -> None:
        callback = _SignalHandlerCallback(self)
        self.trainer.callbacks = self.trainer.callbacks + [callback]

    def register_signal_handlers(self) -> None:
        if _IS_WINDOWS:
            # Windows seems to have signal incompatibilities
            rank_zero_info("Not registering signal handlers on Windows OS")
            return

        if threading.current_thread() is not threading.main_thread():
            # Skip signal registration to allow training in non-main-threads
            rank_zero_debug("Not registering signal handlers outside of the main thread")
            return

        self._register_signal_handler(signal.SIGTERM, self.sigterm_flag)

        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            requeue_signal = environment.requeue_signal
            if requeue_signal is None:
                rank_zero_warn("Requested SLURM auto-requeueing, but signal is disabled. Could not set it up.")
            else:
                rank_zero_info(f"SLURM auto-requeueing enabled. Setting signal handlers for {requeue_signal.name}.")
                self._register_signal_handler(requeue_signal, self.requeue_flag)

    def _process_signals(self) -> None:
        if self.requeue_flag.check_and_reset():
            rank_zero_info("Handling auto-requeue signal")
            self._slurm_requeue()

        if self.sigterm_flag.check_and_reset():
            log.info(rank_prefixed_message("Received SIGTERM. Stopping.", self.trainer.local_rank))
            # Forward signal to subprocesses the first time it is received
            if not self.received_sigterm:
                launcher = self.trainer.strategy.launcher
                if launcher is not None:
                    launcher.kill(signal.SIGTERM)
            self.received_sigterm = True

    def _slurm_requeue(self) -> None:
        # Save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(self.trainer.default_root_dir)
        self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer.is_global_zero:
            job_id = self._slurm_job_id()
            cmd = ["scontrol", "requeue", job_id]

            # Requeue job
            log.info(f"Requeueing job {job_id}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context.
                # Try enlisting the help of the shell to resolve the `scontrol` binary.
                result = subprocess.run(" ".join(cmd), capture_output=True, text=True, shell=True)

            # Print result text
            if result.returncode == 0:
                log.info(f"Requeued SLURM job {job_id}")
            else:
                log.warning(
                    f"Requeueing SLURM job {job_id} failed with error code {result.returncode}: {result.stderr}"
                )

        self.trainer.should_stop = True

    def teardown(self) -> None:
        """Restores the signals that :class:`_SignalConnector` overwrote."""
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._original_handlers.items():
                signal.signal(signum, handler)
        self._original_handlers = {}

    def _register_signal_handler(self, signum: _SIGNUM, handler: _HANDLER) -> None:
        orig_handler = signal.getsignal(signum)
        self._original_handlers[signum] = orig_handler
        signal.signal(signum, _HandlersCompose([orig_handler, handler]))

    def _slurm_job_id(self) -> str:
        array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_job_id is not None:
            array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
            return f"{array_job_id}_{array_task_id}"
        return os.environ["SLURM_JOB_ID"]


def _get_sigkill_signal() -> _SIGNUM:
    return signal.SIGTERM if _IS_WINDOWS else signal.SIGKILL
