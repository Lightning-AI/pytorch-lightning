import os
import signal
from subprocess import call

from pytorch_lightning import _logger as log


class SLURMConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def register_slurm_signal_handlers(self):
        # see if we're using slurm (not interactive)
        on_slurm = False
        try:
            job_name = os.environ['SLURM_JOB_NAME']
            if job_name != 'bash':
                on_slurm = True
        # todo: specify the possible exception
        except Exception:
            pass

        if on_slurm:
            log.info('Set SLURM handle signals.')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        # Todo: required argument `signum` is not used
        # Todo: required argument `frame` is not used
        if self.trainer.is_global_zero:
            # save weights
            log.info('handling SIGUSR1')
            self.trainer.checkpoint_connector.hpc_save(self.trainer.weights_save_path, self.trainer.logger)

            # find job id
            job_id = os.environ['SLURM_JOB_ID']
            cmd = ['scontrol', 'requeue', job_id]

            # requeue job
            log.info(f'requeing job {job_id}...')
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                joint_cmd = [str(x) for x in cmd]
                result = call(' '.join(joint_cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

            # close experiment to avoid issues
            self.trainer.logger.close()

    def term_handler(self, signum, frame):
        # Todo: required argument `signum` is not used
        # Todo: required argument `frame` is not used
        log.info("bypassing sigterm")
