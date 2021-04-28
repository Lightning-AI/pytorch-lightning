from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.optimizer_loop import OptimizerLoop
from pytorch_lightning.utilities import AttributeDict


class BatchLoop(Loop):
    """ Runs over a single batch of data. """

    def on_run_start(self, batch, batch_idx, dataloader_idx):
        self._grad_norm_dic = {}
        self.trainer.hiddens = None
        # self._optimizers = self.prepare_optimizers()
        # lightning module hook
        self._splits = enumerate(self.tbptt_split_batch(batch))
        self.tbptt_loop = OptimizerLoop()

    def on_advance_start(self):
        return super().on_advance_start()

    def advance(self, batch, batch_idx):
        split_idx, split_batch = next(self._splits)
        batch_outputs = self.tbptt_loop.run(split_batch, split_idx, batch_idx)

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            training_step_output_for_epoch_end=batch_outputs,
        )
        return result

    def run(self, batch, batch_idx, dataloader_idx):
        # TODO why is this not in on_run_start?
        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        return super().run(batch, batch_idx, dataloader_idx)

    def tbptt_split_batch(self, batch):
        splits = [batch]
        if self.trainer.truncated_bptt_steps is not None:
            model_ref = self.trainer.lightning_module
            with self.trainer.profiler.profile("tbptt_split_batch"):
                splits = model_ref.tbptt_split_batch(batch, self.trainer.truncated_bptt_steps)
        return splits
