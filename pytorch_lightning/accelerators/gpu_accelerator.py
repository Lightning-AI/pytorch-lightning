try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

from pytorch_lightning.trainer.auto_mix_precision import NATIVE_AMP_AVALAIBLE


class GPUAccelerator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):
        # call setup
        if not self.trainer.testing:
            self.trainer.setup('fit')
            model.setup('fit')

        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        # TODO: remove with dropping NVIDIA AMP support
        if self.trainer.use_amp and not NATIVE_AMP_AVALAIBLE:
            self._setup_nvidia_apex(model)

    def _setup_nvidia_apex(self, model):
        model, optimizers = model.configure_apex(amp, model, self.trainer.optimizers, self.trainer.amp_level)
        self.trainer.optimizers = optimizers
        self.trainer.reinit_scheduler_properties(self.trainer.optimizers, self.trainer.lr_schedulers)
