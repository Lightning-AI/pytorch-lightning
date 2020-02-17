"""
Hooks
=====

There are cases when you might want to do something different at different parts of the training/validation loop.
 To enable a hook, simply override the method in your LightningModule and the trainer will call it at the correct time.

**Contributing** If there's a hook you'd like to add, simply:
1. Fork PyTorchLightning.
2. Add the hook :py:mod:`pytorch_lightning.base_module.hooks.py`.
3. Add the correct place in the :py:mod:`pytorch_lightning.models.trainer` where it should be called.

"""


import torch


try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class ModelHooks(torch.nn.Module):

    def on_sanity_check_start(self):
        """
        Called before starting evaluate
        .. warning:: will be deprecated.
        :return:
        """

    def on_train_start(self):
        """Called at the beginning of training before sanity check
        :return:
        """
        # do something at the start of training

    def on_train_end(self):
        """
        Called at the end of training before logger experiment is closed
        :return:
        """
        # do something at the end of training

    def on_batch_start(self, batch):
        """Called in the training loop before anything happens for that batch.

        :param batch:
        :return:
        """
        # do something when the batch starts

    def on_batch_end(self):
        """Called in the training loop after the batch."""
        # do something when the batch ends

    def on_epoch_start(self):
        """Called in the training loop at the very beginning of the epoch."""
        # do something when the epoch starts

    def on_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        # do something when the epoch ends

    def on_pre_performance_check(self):
        """Called at the very beginning of the validation loop."""
        # do something before validation starts

    def on_post_performance_check(self):
        """Called at the very end of the validation loop."""
        # do something before validation end

    def on_before_zero_grad(self, optimizer):
        """Called after optimizer.step() and before optimizer.zero_grad()

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        for optimizer in optimizers::

            optimizer.step()
            model.on_before_zero_grad(optimizer) # < ---- called here
            optimizer.zero_grad

        :param optimizer:
        :return:
        """
        # do something with the optimizer or inspect it.

    def on_after_backward(self):
        """Called after loss.backward() and before optimizers do anything.

        :return:

        Called in the training loop after model.backward()
        This is the ideal place to inspect or log gradient information

        .. code-block:: python

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

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """Override backward with your own implementation if you need to

        :param trainer: Pointer to the trainer
        :param loss: Loss is already scaled by accumulated grads
        :param optimizer: Current optimizer being used
        :param optimizer_idx: Index of the current optimizer being used
        :return:

        Called to perform backward step.
        Feel free to override as needed.

        The loss passed in has already been scaled for accumulated gradients if requested.

        .. code-block:: python

            def backward(self, use_amp, loss, optimizer):
                if use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        """
        if trainer.precision == 16:

            # .backward is not special on 16-bit with TPUs
            if not trainer.on_tpu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
