from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from pytorch_lightning.utilities import move_data_to_device


try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True


class ModelHooks(Module):

    # TODO: remove in v0.9.0
    def on_sanity_check_start(self):
        """
        Called before starting evaluation.

        Warning:
            Deprecated. Will be removed in v0.9.0.
        """

    def on_train_start(self) -> None:
        """
        Called at the beginning of training before sanity check.
        """
        # do something at the start of training

    def on_train_end(self) -> None:
        """
        Called at the end of training before logger experiment is closed.
        """
        # do something at the end of training

    def on_batch_start(self, batch: Any) -> None:
        """
        Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
        """
        # do something when the batch starts

    def on_batch_end(self) -> None:
        """
        Called in the training loop after the batch.
        """
        # do something when the batch ends

    def on_epoch_start(self) -> None:
        """
        Called in the training loop at the very beginning of the epoch.
        """
        # do something when the epoch starts

    def on_epoch_end(self) -> None:
        """
        Called in the training loop at the very end of the epoch.
        """
        # do something when the epoch ends

    def on_pre_performance_check(self) -> None:
        """
        Called at the very beginning of the validation loop.
        """
        # do something before validation starts

    def on_post_performance_check(self) -> None:
        """
        Called at the very end of the validation loop.
        """
        # do something before validation end

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Called after optimizer.step() and before optimizer.zero_grad().

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        This is where it is called::

            for optimizer in optimizers:
                optimizer.step()
                model.on_before_zero_grad(optimizer) # < ---- called here
                optimizer.zero_grad

        Args:
            optimizer: The optimizer for which grads should be zeroed.
        """
        # do something with the optimizer or inspect it.

    def on_after_backward(self) -> None:
        """
        Called in the training loop after loss.backward() and before optimizers do anything.
        This is the ideal place to inspect or log gradient information.

        Example::

            def on_after_backward(self):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    params = self.state_dict()
                    for k, v in params.items():
                        grads = v
                        name = k
                        self.logger.experiment.add_histogram(tag=name, values=grads,
                                                             global_step=self.trainer.global_step)

        """

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        """
        Override backward with your own implementation if you need to.

        Args:
            trainer: Pointer to the trainer
            loss: Loss is already scaled by accumulated grads
            optimizer: Current optimizer being used
            optimizer_idx: Index of the current optimizer being used

        Called to perform backward step.
        Feel free to override as needed.

        The loss passed in has already been scaled for accumulated gradients if requested.

        Example::

            def backward(self, use_amp, loss, optimizer):
                if use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        """
        if trainer.precision == 16:
            # .backward is not special on 16-bit with TPUs
            if trainer.on_tpu:
                return

            if self.trainer.use_native_amp:
                self.trainer.scaler.scale(loss).backward()

            # TODO: remove in v0.8.0
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors
        wrapped in a custom data structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`
        - ``torchtext.data.Batch`` (COMING SOON)

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Example::

            def transfer_batch_to_device(self, batch, device)
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                else:
                    batch = super().transfer_batch_to_device(data, device)
                return batch

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.

        Returns:
            A reference to the data on the new device.

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).
            The :class:`~pytorch_lightning.trainer.trainer.Trainer` already takes care of splitting the
            batch and determines the target devices.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
            - :func:`~pytorch_lightning.utilities.apply_func.apply_to_collection`
        """
        return move_data_to_device(batch, device)
