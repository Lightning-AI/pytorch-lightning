import torch

from pytorch_lightning.callbacks import GradientAccumulationScheduler


class TrainerTrainingTricksMixin(object):

    def clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def print_nan_gradients(self):
        model = self.get_model()
        for param in model.parameters():
            if torch.isnan(param.grad.float()).any():
                print(param, param.grad)

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None

        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")
