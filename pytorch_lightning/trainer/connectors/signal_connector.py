import logging
import os
import signal
from subprocess import call

from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities.imports import _fault_tolerant_training

log = logging.getLogger(__name__)


class SignalConnector:
    def __init__(self, trainer):
        self.trainer = trainer
        self.trainer._should_gracefully_terminate = False

    def register_signal_handlers(self):
        cluster_env = getattr(self.trainer.training_type_plugin, "cluster_environment", None)
        if isinstance(cluster_env, SLURMEnvironment):
            self.register_slurm_signal_handlers()
        elif _fault_tolerant_training():
            self.register_fault_tolerant_handlers()

    def register_fault_tolerant_handlers(self):
        signal.signal(signal.SIGUSR1, self.sig_fault_tolerant_handler)
        signal.signal(signal.SIGTERM, self.term_handler)

    def register_slurm_signal_handlers(self):
        # see if we're using slurm (not interactive)
        on_slurm = False
        try:
            job_name = os.environ["SLURM_JOB_NAME"]
            if job_name != "bash":
                on_slurm = True
        # todo: specify the possible exception
        except Exception:
            pass

        if on_slurm:
            log.info("Set SLURM handle signals.")
            signal.signal(signal.SIGUSR1, self.sig_slurm_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_slurm_handler(self, signum, frame):  # pragma: no-cover
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

    def sig_fault_tolerant_handler(self, signum, frame):  # pragma: no-cover
        self.trainer._should_gracefully_terminate = True

    def term_handler(self, signum, frame):  # pragma: no-cover
        log.info("bypassing sigterm")
