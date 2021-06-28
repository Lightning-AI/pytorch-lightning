from pytorch_lightning.callbacks import Callback


class AutoRestart(Callback):

    def __init__(
        self,
        save_gradients_on_accumulate_grad_batches: bool = True,
    ):
        self.save_gradients_on_accumulate_grad_batches = save_gradients_on_accumulate_grad_batches

    def
