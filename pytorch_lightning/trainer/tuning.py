from pytorch_lightning.trainer.batch_size_scaling import scale_batch_size


class Tuner:

    def __init__(self, trainer):
        self.trainer = trainer

    def scale_batch_size(self,
                         model,
                         mode: str = 'power',
                         steps_per_trial: int = 3,
                         init_val: int = 2,
                         max_trials: int = 25,
                         batch_arg_name: str = 'batch_size',
                         **fit_kwargs):
        return scale_batch_size(
            self.trainer, model, mode, steps_per_trial, init_val, max_trials, batch_arg_name, **fit_kwargs
        )


