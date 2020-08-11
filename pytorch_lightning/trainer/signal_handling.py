import atexit
import os
import signal
from subprocess import call

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.trainer.states import TrainerState


class SignalHandler:
    """
    Takes care of registering and restoring signal handlers for the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`. This includes handling
    graceful shutdown for KeyboardInterrupt or SLURM autoresubmit signals.
    """

    def __init__(self, trainer: "pl.Trainer"):
        self.trainer = trainer
        self.original_handlers = {}

    def register(self):
        """
        Registers the signal handlers for the Trainer, including
        - training teardown signal handlers that run on interpreter exit and other POSIX signals.
        - HPC signal handling, i.e., registering auto-resubmit when on SLURM
        """
        self._register_signal(signal.SIGTERM, self._teardown_handler)
        self._register_signal(signal.SIGSEGV, self._teardown_handler)
        self._register_signal(signal.SIGINT, self._teardown_handler)

        if on_slurm():
            log.info('Set SLURM handle signals.')
            self._register_signal(signal.SIGUSR1, self._slurm_auto_resubmit_handler)
            self._register_signal(signal.SIGTERM, self._slurm_sigterm_handler)

        # atexit.register(self.trainer.run_training_teardown)

    def restore(self):
        """ Restores the original signal handlers (e.g. the Python defaults) """
        for signum, handler in self.original_handlers.items():
            signal.signal(signum, handler)

    def _register_signal(self, signum, handler):
        """ Registers a signal handler and saves a reference to the original handler. """
        self.original_handlers.update({signum: signal.getsignal(signum)})
        signal.signal(signum, handler)

    def _slurm_auto_resubmit_handler(self, signum, frame):  # pragma: no-cover
        """ This handler resubmits the SLURM job when SIGUSR1 is signaled. """
        trainer = self.trainer
        if trainer.is_global_zero:
            # save weights
            log.info('handling SIGUSR1')
            trainer.hpc_save(trainer.weights_save_path, trainer.logger)

            # find job id
            job_id = os.environ['SLURM_JOB_ID']
            cmd = 'scontrol requeue {}'.format(job_id)

            # requeue job
            log.info(f'requeing job {job_id}...')
            result = call(cmd, shell=True)

            # print result text
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

            # close experiment to avoid issues
            trainer.logger.close()

    def _slurm_sigterm_handler(self, signum, frame):  # pragma: no-cover
        # TODO: Implement this. Currently will just block the process
        log.info("bypassing sigterm")

    def _teardown_handler(self, signum, frame):  # pragma: no-cover
        """ Handles training teardown for certain signals that interrupt training. """
        trainer = self.trainer
        if not trainer.interrupted:
            trainer.interrupted = True
            trainer.state = TrainerState.INTERRUPTED
            trainer.on_keyboard_interrupt()
            trainer.run_training_teardown()
            raise KeyboardInterrupt


def on_slurm() -> bool:  # pragma: no-cover
    """ Checks if we're using SLURM (not interactive). """
    job_name = os.environ.get('SLURM_JOB_NAME')
    return job_name is not None and job_name != 'bash'
