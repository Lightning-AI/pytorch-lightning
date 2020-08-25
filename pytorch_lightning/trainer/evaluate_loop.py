import torch
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.core.step_result import EvalResult


class EvaluationLoop(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.testing = False
        self.outputs = []
        self.predictions = None
        self.max_batches = None

    def is_using_eval_results(self):
        outputs = self.outputs
        using_eval_result = len(outputs) > 0 and len(outputs[0]) > 0 and isinstance(outputs[0][0], EvalResult)
        return using_eval_result

    def setup(self, model, max_batches, dataloaders):
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.trainer.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches

    def on_evaluation_epoch_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def evaluation_step(self, *args, **kwargs):
        if self.testing:
            output = self.trainer.accelerator_backend.test_step(*args, **kwargs)
        else:
            output = self.trainer.accelerator_backend.validation_step(*args, **kwargs)
        return output

    def evaluation_step_end(self, *args, **kwargs):
        if self.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def on_evaluation_batch_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_batch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_batch_start', *args, **kwargs)

    def on_evaluation_batch_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_batch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_batch_end', *args, **kwargs)

    def on_evaluation_epoch_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_epoch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_end', *args, **kwargs)