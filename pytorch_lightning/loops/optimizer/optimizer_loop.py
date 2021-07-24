from collections import defaultdict
from typing import Any, Dict, Optional

import torch

from pytorch_lightning.loops import Loop
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.utilities import AttributeDict


class OptimizerLoop(Loop):

    def __init__(self):
        super().__init__()
        self.outputs: Dict[int, Optional[AttributeDict]] = defaultdict(None)
        self.optim_progress = OptimizationProgress()

    @property
    def done(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:

    # for opt_idx, optimizer in self.get_active_optimizers(batch_idx):
    # handle optimization restart
    # if self.restarting:
    #     if opt_idx < self.optim_progress.optimizer_idx:
    #         continue

        # self.optim_progress.optimizer_idx = opt_idx

        result = self._run_optimization(batch_idx, split_batch, opt_idx, optimizer)
        if result:
            self.outputs[self.optim_progress.optimizer_idx].append(result.training_step_output)

    def _run_optimization(
        self, batch_idx: int, split_batch: Any, opt_idx: int = 0, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Runs closure (train step + backward) together with optimization if necessary.

        Args:
            batch_idx: the index of the current batch
            split_batch: the current tbptt split of the whole batch
            opt_idx: the index of the current optimizer
            optimizer: the current optimizer
        """
        # TODO(@awaelchli): In v1.5, when optimizer_idx gets removed from training_step in manual_optimization, change
        #   opt_idx=0 to opt_idx=None in the signature here

        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)

        result = AttributeDict()
        closure = self._make_closure(split_batch, batch_idx, opt_idx, optimizer, self._hiddens, result)

        if self.should_accumulate():
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            # automatic_optimization=False: don't block synchronization here
            with self.block_ddp_sync_behaviour():
                closure()

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            if self.trainer.lightning_module.automatic_optimization:
                self._optimizer_step(optimizer, opt_idx, batch_idx, closure)
            else:
                result = self._training_step(split_batch, batch_idx, opt_idx, self._hiddens)

        if result:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            self._update_running_loss(result.loss)
            self._process_closure_result(result)

        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result