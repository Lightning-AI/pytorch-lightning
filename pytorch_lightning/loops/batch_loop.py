from pytorch_lightning.loops.base import Loop
from pytorch_lightning.utilities import AttributeDict


class BatchLoop(Loop):
    """ Runs over a single batch of data. """

    def on_run_start(self, batch, batch_idx, dataloader_idx):
        self._grad_norm_dic = {}
        self.trainer.hiddens = None
        self._optimizers = self.prepare_optimizers()
        # lightning module hook
        self._splits = enumerate(self.tbptt_split_batch(batch))
        self.tbptt_loop = BatchSplitLoop(self._optimizers)

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


class BatchSplitLoop(Loop):
    """ Runs over a single split of a batch of data (TBPTT). """

    def __init__(self, optimizers):
        super().__init__()
        self._optimizers = enumerate(optimizers)

    def advance(self, split_batch, split_idx, batch_idx):
        opt_idx, optimizer = next(self._optimizers)

        # toggle model params + set info to logger_connector
        self.run_train_split_start(split_idx, split_batch, opt_idx, optimizer)

        if self.should_accumulate():
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------

            # automatic_optimization=True: perform dpp sync only when performing optimizer_step
            # automatic_optimization=False: don't block synchronization here
            with self.block_ddp_sync_behaviour():
                self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens)

            batch_outputs = self._process_closure_result(
                batch_outputs=batch_outputs,
                opt_idx=opt_idx,
            )

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients

        else:
            if self.automatic_optimization:

                def train_step_and_backward_closure():
                    result = self.training_step_and_backward(
                        split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                    )
                    return None if result is None else result.loss

                # optimizer step
                self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

            else:
                self._curr_step_result = self.training_step(split_batch, batch_idx, opt_idx, self.trainer.hiddens)

            if self._curr_step_result is None:
                # user decided to skip optimization
                # make sure to zero grad.
                # TODO add logic to skip in the outer loop
                return
                # continue

            batch_outputs = self._process_closure_result(
                batch_outputs=batch_outputs,
                opt_idx=opt_idx,
            )

            # todo: Properly aggregate grad_norm accros opt_idx and split_idx
            grad_norm_dic = self._cur_grad_norm_dict
            self._cur_grad_norm_dict = None

            # update running loss + reset accumulated loss
            self.update_running_loss()